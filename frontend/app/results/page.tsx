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
        const latParam = searchParams.get("lat");
        const lngParam = searchParams.get("lng");

        if (!latParam || !lngParam) {
          setError("Location parameters are missing.");
          setLoading(false);
          return;
        }

        const latitude = parseFloat(latParam);
        const longitude = parseFloat(lngParam);

        const headers = new Headers();
        headers.append("Content-Type", "application/json");

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

        const recommendedCrops: string[] = cropData.top3_crops;
        if (!recommendedCrops || recommendedCrops.length === 0) {
          throw new Error("No crops were returned from the backend.");
        }

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
          console.log(yieldData); // Debugging line
          return {
            name: yieldData.type || crop, 
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

  return (
    <div className="flex flex-col min-h-screen">
      {/* Main content area */}
      <main className="flex-grow flex items-center justify-center pb-10">
        {loading ? (
          // Loader: Using an animated spinner (TailwindCSS)
          <div className="flex flex-col items-center justify-center">
            <svg
              className="animate-spin h-12 w-12 text-green-600 mb-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              ></path>
            </svg>
            <p className="text-xl text-green-600">Loading...</p>
          </div>
        ) : error ? (
          // Error message
          <div className="text-center text-red-600">
            <p>{error}</p>
          </div>
        ) : (
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
            <div className="flex flex-col items-center justify-center py-2">
              <h2 className="text-3xl font-bold mb-4 text-green-700">
              Croptimizer: Top 6 Recommended Crops
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
              {crops.map((crop, index) => (
                <div key={index} className="bg-white shadow-lg rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-2">{crop.name}</h3>
                <p className="text-gray-600">{crop.yield}</p>
                </div>
              ))}
              </div>
            </div>
            </div>
        )}
      </main>
    </div>
  );
}
