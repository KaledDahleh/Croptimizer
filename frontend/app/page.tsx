import Link from "next/link";

export default function Home() {
  return (
    <div className="relative min-h-screen">
      <div
        className="absolute inset-0 w-full h-full bg-cover bg-center z-0"
        style={{ backgroundImage: "url('/farm.png')" }}
      >
        <div className="absolute inset-0 bg-black opacity-40"></div>
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen py-2 text-white">
        <h1 className="text-4xl font-bold ml-5 md:ml-0 mb-8">Welcome to Croptimizer</h1>
        <p className="p-5 text-md mb-8 text-left md:text-center max-w-4/5">
          Optimize your crop selection based on your location, soil type, and preferences. Let's make agriculture more
          efficient and sustainable together!
        </p>
        <Link
          href="/location"
          className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
        >
          Get Started
        </Link>
      </div>
    </div>
  );
}
