"use client"

import { useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"

const soilTypes = ["Clay", "Sandy", "Loamy"]
const plantTypes = ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", "Mungbean", "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee"]

export default function SoilAndPlant() {
  const [soilType, setSoilType] = useState("")
  const [plantType, setPlantType] = useState("")
  const router = useRouter()
  const searchParams = useSearchParams()

  const handleSubmit = (e) => {
    e.preventDefault()
    const lat = searchParams.get("lat")
    const lng = searchParams.get("lng")
    router.push(`/results?lat=${lat}&lng=${lng}&soil=${soilType}&plant=${plantType}`)
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h2 className="text-3xl font-bold mb-4 text-green-700">Crop-timizer: Select Soil and Plant Type</h2>
      <form onSubmit={handleSubmit} className="w-full max-w-md">
        <div className="mb-4">
          <label htmlFor="soilType" className="block text-sm font-medium text-gray-700">
            Soil Type
          </label>
          <select
            id="soilType"
            value={soilType}
            onChange={(e) => setSoilType(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm rounded-md"
            required
          >
            <option value="">Select a soil type</option>
            {soilTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>
        <div className="mb-4">
          <label htmlFor="plantType" className="block text-sm font-medium text-gray-700">
            Plant Type
          </label>
          <select
            id="plantType"
            value={plantType}
            onChange={(e) => setPlantType(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm rounded-md"
            required
          >
            <option value="">Select a plant type</option>
            {plantTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
          disabled={!soilType || !plantType}
        >
          Get Results
        </button>
      </form>
    </div>
  )
}

