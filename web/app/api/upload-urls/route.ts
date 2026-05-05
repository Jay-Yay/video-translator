import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"
import { NextRequest, NextResponse } from "next/server"

const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
  },
})

export async function POST(request: NextRequest) {
  const { batch_id, filenames } = (await request.json()) as {
    batch_id: string
    filenames: string[]
  }

  if (!batch_id || !Array.isArray(filenames) || filenames.length === 0) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 })
  }

  const urls = await Promise.all(
    filenames.map(async (filename) => {
      const stem = filename.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_\-]/g, "_")
      const key = `batches/${batch_id}/${stem}.mp4`
      const url = await getSignedUrl(
        r2,
        new PutObjectCommand({ Bucket: process.env.R2_BUCKET!, Key: key }),
        { expiresIn: 900 },
      )
      return { filename, stem, r2_key: key, upload_url: url }
    }),
  )

  return NextResponse.json({ urls })
}
