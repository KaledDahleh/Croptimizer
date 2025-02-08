import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const apiKey = import.meta.env.VITE_MAPS_API_KEY;

function Map({ location = "UIC_ARC+IL" }: { location?: string }) {
  return (
    <div className="w-full h-96 md:h-120 bg-gray-300 rounded-lg mt-4">
      <iframe
        src={`https://www.google.com/maps/embed/v1/place?key=${apiKey}&q=${location || "UIC_ARC+IL"}`}
        className="w-full h-full"
        allowFullScreen
      ></iframe>
    </div>
  );
}

function getCurrentLocation(setLocation: (location: string) => void) {
  navigator.geolocation.getCurrentPosition(
    (position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      const location = `${lat},${lon}`;
      setLocation(location);
    },
    (error) => {
      console.error("Error getting location: ", error);
      alert("Unable to retrieve your location. Please check your location settings and permissions.");
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
}

export default function Maps() {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [location, setLocation] = useState<string>("UIC_ARC+IL");

  useEffect(() => {
    getCurrentLocation(setLocation);
  }, []);

  return (
    
    <div className="p-4 m-0 w-full flex flex-col items-center">
      <Map location={location} />

      <div className="w-full flex flex-col md:flex-row md:justify-between">

        <div className="w-full md:w-1/3 bg-white drop-shadow-md p-4 rounded-lg mt-4">
          <h1 className="text-2xl font-bold">Search for a Location</h1>
          <input
            type="text"
            placeholder="Location"
            className="w-full p-2 rounded-lg mt-2"
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <div className="flex flex-row justify-between mt-2">
            <button
              className="bg-blue-500 text-white p-2 rounded-lg mt-2"
              onClick={() => setLocation(searchQuery)}
            >
              Search
            </button>
            <button
              className="bg-red-500 text-white p-2 rounded-lg mt-2"
              onClick={() => getCurrentLocation(setLocation)}
            >
              Use Current Location
            </button>
          </div>
        </div>

      </div>

    </div>
  );
}