import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Video Translator",
  description: "Korean → Japanese video translation",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, background: "#f5f5f5" }}>
        {children}
      </body>
    </html>
  )
}
