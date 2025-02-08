"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import dynamic from "next/dynamic"
import { Button } from "@/components/ui/button"

const MapWithNoSSR = dynamic(() => import("./Map"), {
  ssr: false,
})

export default function Location() {
  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lng: number } | null>(null)
  const router = useRouter()

  const handleLocationSelect = (location: { lat: number; lng: number }) => {
    setSelectedLocation(location)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (selectedLocation) {
      router.push(`/soil-and-plant?lat=${selectedLocation.lat}&lng=${selectedLocation.lng}`)
    }
  }

  const handleCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          }
          setSelectedLocation(location)
        },
        (error) => {
          console.error("Error getting current location:", error)
          alert("Unable to retrieve your location. Please select manually.")
        },
      )
    } else {
      alert("Geolocation is not supported by your browser. Please select location manually.")
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h2 className="text-3xl font-bold mb-4 text-green-700">Select Your Location</h2>
      <div className="w-full max-w-2xl h-96 mb-4">
        <MapWithNoSSR onLocationSelect={handleLocationSelect} selectedLocation={selectedLocation} />
      </div>
      <div className="w-full max-w-2xl flex justify-between mb-4">
        <Button onClick={handleCurrentLocation} variant="outline">
          Use Current Location
        </Button>
        <Button onClick={handleSubmit} disabled={!selectedLocation}>
          Next
        </Button>
      </div>
    </div>
  )
}

