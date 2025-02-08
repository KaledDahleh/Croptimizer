"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

interface Crop {
  name: string;
  yield: number;
}

export default function Results() {
  const [crops, setCrops] = useState<Crop[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const searchParams = useSearchParams();

  useEffect(() => {
    const fetchCropsAndYields = async () => {
      try {
        // Get location from URL search parameters
        const latParam = searchParams.get("lat");
        const lngParam = searchParams.get("lng");

        if (!latParam || !lngParam) {
          setError("Location parameters are missing.");
          setLoading(false);
          return;
        }

        const latitude = parseFloat(latParam);
        const longitude = parseFloat(lngParam);

        // Prepare headers for JSON requests
        const headers = new Headers();
        headers.append("Content-Type", "application/json");

        // --- Step 1: Get Top 3 Recommended Crops ---
        const cropRequestOptions: RequestInit = {
          method: "POST", // Changed from GET to POST
          headers,
          body: JSON.stringify({ latitude, longitude }),
          redirect: "follow",
        };

        const cropResponse = await fetch(
          "http://127.0.0.1:5000/predict_crops",
          cropRequestOptions
        );
        if (!cropResponse.ok) {
          throw new Error("Failed to fetch recommended crops");
        }
        const cropData = await cropResponse.json();

        // Expecting response shape: { top3_crops: ["Crop1", "Crop2", "Crop3"] }
        const recommendedCrops: string[] = cropData.top3_crops;
        if (!recommendedCrops || recommendedCrops.length === 0) {
          throw new Error("No crops were returned from the backend.");
        }

        // --- Step 2: For each recommended crop, get its yield ---
        const yieldPromises = recommendedCrops.map(async (crop) => {
          const yieldRequestOptions: RequestInit = {
            method: "POST", // Changed from GET to POST
            headers,
            body: JSON.stringify({
              crop_type: crop,
              latitude,
              longitude,
            }),
            redirect: "follow",
          };

          const yieldResponse = await fetch(
            "http://127.0.0.1:5000/predict_yield",
            yieldRequestOptions
          );
          if (!yieldResponse.ok) {
            throw new Error(`Failed to fetch yield for ${crop}`);
          }
          const yieldData = await yieldResponse.json();
          // Expecting response shape: { type: "CropName", predicted_yield: yieldValue }
          return {
            name: yieldData.type,
            yield: yieldData.predicted_yield,
          };
        });

        const cropsWithYields = await Promise.all(yieldPromises);
        setCrops(cropsWithYields);
      } catch (err) {
        console.error(err);
        setError("Failed to fetch crop data. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchCropsAndYields();
  }, [searchParams]);

  if (loading) {
    return <div className="text-center">Loading...</div>;
  }

  if (error) {
    return <div className="text-center text-red-600">{error}</div>;
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h2 className="text-3xl font-bold mb-4 text-green-700">
        Crop-timizer: Top 3 Recommended Crops
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {crops.map((crop, index) => (
          <div key={index} className="bg-white shadow-lg rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-2">{crop.name}</h3>
            <p className="text-gray-600">
              Predicted Yield: {crop.yield} kg/hectare
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
