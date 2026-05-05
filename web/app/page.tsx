"use client"
import { useCallback, useRef, useState } from "react"

type FileEntry = {
  file: File
  stem: string
  r2_key: string
  upload_url: string
  progress: number  // 0–100
  done: boolean
  error: string | null
}

function generateBatchId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7)
}

export default function UploadPage() {
  const [entries, setEntries] = useState<FileEntry[]>([])
  const [email, setEmail] = useState("")
  const [phase, setPhase] = useState<"idle" | "preparing" | "uploading" | "submitted" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")
  const batchIdRef = useRef(generateBatchId())

  const setEntry = (stem: string, patch: Partial<FileEntry>) =>
    setEntries(prev => prev.map(e => e.stem === stem ? { ...e, ...patch } : e))

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".mp4"))
    if (files.length === 0) return
    setEntries(prev => {
      const existing = new Set(prev.map(e => e.file.name))
      const fresh = files.filter(f => !existing.has(f.name))
      return [...prev, ...fresh.map(f => ({
        file: f,
        stem: f.name.replace(/\.mp4$/i, "").replace(/[^a-zA-Z0-9_\-]/g, "_"),
        r2_key: "",
        upload_url: "",
        progress: 0,
        done: false,
        error: null,
      }))]
    })
  }, [])

  async function handleUpload() {
    if (!email || entries.length === 0) return
    setPhase("preparing")

    const res = await fetch("/api/upload-urls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_id: batchIdRef.current,
        filenames: entries.map(e => e.file.name),
      }),
    })
    if (!res.ok) { setPhase("error"); setErrorMsg("Failed to get upload URLs"); return }
    const { urls } = await res.json()

    setEntries(prev => prev.map(e => {
      const match = urls.find((u: { stem: string }) => u.stem === e.stem)
      return match ? { ...e, r2_key: match.r2_key, upload_url: match.upload_url } : e
    }))

    setPhase("uploading")

    await Promise.all(
      entries.map(async (entry, i) => {
        const urlEntry = urls[i]
        try {
          await new Promise<void>((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.upload.onprogress = ev => {
              if (ev.lengthComputable)
                setEntry(entry.stem, { progress: Math.round((ev.loaded / ev.total) * 100) })
            }
            xhr.onload = () => xhr.status < 300 ? resolve() : reject(new Error(`HTTP ${xhr.status}`))
            xhr.onerror = () => reject(new Error("Network error"))
            xhr.open("PUT", urlEntry.upload_url)
            xhr.setRequestHeader("Content-Type", "video/mp4")
            xhr.send(entry.file)
          })
          setEntry(entry.stem, { done: true, progress: 100 })
        } catch (err) {
          setEntry(entry.stem, { error: String(err) })
        }
      })
    )

    const submitRes = await fetch("/api/submit-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_id: batchIdRef.current,
        videos: entries.map(e => ({ r2_key: urls.find((u: { stem: string }) => u.stem === e.stem)?.r2_key, stem: e.stem })),
        notify_email: email,
      }),
    })
    if (!submitRes.ok) { setPhase("error"); setErrorMsg("Failed to submit batch"); return }
    setPhase("submitted")
  }

  if (phase === "submitted") {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh" }}>
        <div style={{ textAlign: "center", background: "white", padding: 40, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}>
          <div style={{ fontSize: 48 }}>&#10003;</div>
          <h2>Submitted!</h2>
          <p>You&apos;ll receive an email at <strong>{email}</strong> when your {entries.length} video{entries.length > 1 ? "s are" : " is"} ready.</p>
          <p style={{ color: "#666", fontSize: 14 }}>Results will appear in Google Drive &rsaquo; KR&#x2192;JP Translations</p>
          <button onClick={() => { setPhase("idle"); setEntries([]); batchIdRef.current = generateBatchId() }}
            style={{ marginTop: 16, padding: "10px 24px", background: "#0070f3", color: "white", border: "none", borderRadius: 4, fontSize: 16, cursor: "pointer" }}>
            Upload another batch
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ marginBottom: 8 }}>KR &rarr; JP Video Translator</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>Upload Korean beauty videos. Translated Premiere Pro projects will land in Google Drive.</p>

      <div
        onDrop={onDrop}
        onDragOver={e => e.preventDefault()}
        style={{ border: "2px dashed #ccc", borderRadius: 8, padding: 40, textAlign: "center", color: "#666", marginBottom: 24, cursor: "pointer", background: "white" }}
      >
        <div style={{ fontSize: 32, marginBottom: 8 }}>&#8679;</div>
        Drag &amp; drop <strong>.mp4</strong> files here
        <div style={{ marginTop: 12 }}>
          <label style={{ cursor: "pointer", color: "#0070f3" }}>
            or click to browse
            <input type="file" accept=".mp4" multiple style={{ display: "none" }}
              onChange={e => {
                const files = Array.from(e.target.files || [])
                setEntries(prev => {
                  const existing = new Set(prev.map(en => en.file.name))
                  return [...prev, ...files.filter(f => !existing.has(f.name)).map(f => ({
                    file: f, stem: f.name.replace(/\.mp4$/i, "").replace(/[^a-zA-Z0-9_\-]/g, "_"),
                    r2_key: "", upload_url: "", progress: 0, done: false, error: null,
                  }))]
                })
              }}
            />
          </label>
        </div>
      </div>

      {entries.length > 0 && (
        <div style={{ background: "white", borderRadius: 8, padding: 16, marginBottom: 24 }}>
          {entries.map(e => (
            <div key={e.stem} style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontWeight: 500 }}>{e.file.name}</span>
                <span style={{ color: "#666", fontSize: 14 }}>{(e.file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <div style={{ height: 6, background: "#eee", borderRadius: 3 }}>
                <div style={{ height: "100%", background: e.error ? "#e00" : e.done ? "#0a0" : "#0070f3", borderRadius: 3, width: `${e.progress}%`, transition: "width 0.3s" }} />
              </div>
              {e.error && <div style={{ color: "#e00", fontSize: 13, marginTop: 4 }}>{e.error}</div>}
            </div>
          ))}
        </div>
      )}

      <div style={{ marginBottom: 16 }}>
        <label style={{ display: "block", marginBottom: 6, fontWeight: 500 }}>Notify email</label>
        <input
          type="email"
          placeholder="your@email.com"
          value={email}
          onChange={e => setEmail(e.target.value)}
          style={{ width: "100%", padding: "10px 12px", fontSize: 16, border: "1px solid #ccc", borderRadius: 4, boxSizing: "border-box" }}
        />
      </div>

      {phase === "error" && <p style={{ color: "red" }}>{errorMsg}</p>}
      <button
        onClick={handleUpload}
        disabled={entries.length === 0 || !email || phase === "uploading" || phase === "preparing"}
        style={{
          width: "100%", padding: "12px 0", fontSize: 16,
          background: entries.length === 0 || !email ? "#ccc" : "#0070f3",
          color: "white", border: "none", borderRadius: 4, cursor: entries.length === 0 || !email ? "not-allowed" : "pointer",
        }}
      >
        {phase === "preparing" ? "Preparing uploads…" :
         phase === "uploading" ? `Uploading… (${entries.filter(e => e.done).length}/${entries.length})` :
         `Translate ${entries.length > 0 ? entries.length : ""} video${entries.length !== 1 ? "s" : ""}`}
      </button>
    </div>
  )
}
