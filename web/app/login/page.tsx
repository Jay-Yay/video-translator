"use client"
import { useState } from "react"
import { useRouter } from "next/navigation"

export default function LoginPage() {
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const router = useRouter()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    })
    if (res.ok) {
      router.push("/")
    } else {
      setError("Incorrect password")
    }
  }

  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh" }}>
      <form onSubmit={handleSubmit} style={{ background: "white", padding: 32, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.1)", width: 320 }}>
        <h2 style={{ margin: "0 0 24px" }}>Video Translator</h2>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          style={{ width: "100%", padding: "10px 12px", fontSize: 16, border: "1px solid #ccc", borderRadius: 4, boxSizing: "border-box" }}
          autoFocus
        />
        {error && <p style={{ color: "red", margin: "8px 0 0" }}>{error}</p>}
        <button
          type="submit"
          style={{ marginTop: 16, width: "100%", padding: "10px 0", background: "#0070f3", color: "white", border: "none", borderRadius: 4, fontSize: 16, cursor: "pointer" }}
        >
          Sign in
        </button>
      </form>
    </div>
  )
}
