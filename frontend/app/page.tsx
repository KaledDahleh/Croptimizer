import Link from "next/link"

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-4xl font-bold mb-8 text-green-700">Welcome to Crop-timizer</h1>
      <p className="text-xl mb-8 text-center max-w-2xl">
        Optimize your crop selection based on your location, soil type, and preferences. Let's make agriculture more
        efficient and sustainable together!
      </p>
      <Link href="/location" className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
        Get Started
      </Link>
    </div>
  )
}

