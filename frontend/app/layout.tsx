import "./globals.css"
import { Inter } from "next/font/google"
import type React from "react" 

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Croptimizer",
  description: "Optimize your crop selection based on your location",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <header className="bg-green-600 text-white p-4">
          <h1 className="text-2xl font-bold">Croptimizer</h1>
        </header>
        <main className="container mx-auto p-4">{children}</main>
        <footer className="bg-green-600 text-white p-4 mt-8">
          <p>&copy; 2025 Croptimizer</p>
        </footer>
      </body>
    </html>
  )
}

