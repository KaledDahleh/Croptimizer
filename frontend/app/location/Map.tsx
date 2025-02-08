"use client"

import { useEffect, useRef, useCallback } from "react"
import { Loader } from "@googlemaps/js-api-loader"

interface MapProps {
  onLocationSelect: (location: { lat: number; lng: number }) => void
  selectedLocation: { lat: number; lng: number } | null
}

const Map: React.FC<MapProps> = ({ onLocationSelect, selectedLocation }) => {
  const mapRef = useRef<HTMLDivElement>(null)
  const googleMapRef = useRef<google.maps.Map | null>(null)
  const markerRef = useRef<google.maps.Marker | null>(null)

  const handleMapClick = useCallback(
    (e: google.maps.MapMouseEvent) => {
      if (e.latLng) {
        onLocationSelect({ lat: e.latLng.lat(), lng: e.latLng.lng() })
      }
    },
    [onLocationSelect],
  )

  useEffect(() => {
    const loader = new Loader({
      apiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY!,
      version: "weekly",
    })

    loader.load().then(() => {
      if (mapRef.current && !googleMapRef.current) {
        const map = new google.maps.Map(mapRef.current, {
          center: { lat: 0, lng: 0 },
          zoom: 2,
        })
        googleMapRef.current = map
        map.addListener("click", handleMapClick)
      }
    })
  }, [handleMapClick])

  useEffect(() => {
    if (googleMapRef.current && selectedLocation) {
      if (markerRef.current) {
        markerRef.current.setMap(null)
      }
      markerRef.current = new google.maps.Marker({
        position: selectedLocation,
        map: googleMapRef.current,
      })
      googleMapRef.current.panTo(selectedLocation)
      googleMapRef.current.setZoom(10)
    }
  }, [selectedLocation])

  return <div ref={mapRef} style={{ width: "100%", height: "100%" }} />
}

export default Map

