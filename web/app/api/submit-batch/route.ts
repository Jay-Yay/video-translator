import { Redis } from "@upstash/redis"
import { NextRequest, NextResponse } from "next/server"

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
})

interface VideoEntry {
  r2_key: string
  stem: string
}

export async function POST(request: NextRequest) {
  const { batch_id, videos, notify_email } = (await request.json()) as {
    batch_id: string
    videos: VideoEntry[]
    notify_email: string
  }

  if (!batch_id || !videos?.length || !notify_email) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 })
  }

  // Write batch record to Redis (TTL: 7 days)
  const ttl = 60 * 60 * 24 * 7
  await Promise.all([
    redis.set(`batch:${batch_id}:total`, videos.length, { ex: ttl }),
    redis.set(`batch:${batch_id}:completed`, 0, { ex: ttl }),
    redis.set(`batch:${batch_id}:done`, 0, { ex: ttl }),
    redis.set(`batch:${batch_id}:email`, notify_email, { ex: ttl }),
  ])

  // Trigger Modal dispatch endpoint
  const modalRes = await fetch(process.env.MODAL_DISPATCH_URL!, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Dispatch-Secret": process.env.MODAL_DISPATCH_SECRET!,
    },
    body: JSON.stringify({ batch_id, videos }),
  })

  if (!modalRes.ok) {
    const text = await modalRes.text()
    return NextResponse.json(
      { error: `Modal dispatch failed: ${modalRes.status} ${text}` },
      { status: 502 },
    )
  }

  return NextResponse.json({ ok: true, batch_id, count: videos.length })
}
