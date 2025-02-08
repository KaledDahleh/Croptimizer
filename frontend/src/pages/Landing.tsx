import { Link } from 'react-router-dom';

const Landing = () => {
    return (


        <div className="h-screen flex flex-col items-center justify-center bg-green-500">

            <div className="w-1/2 bg-white opacity-90 p-8 drop-shadow-lg rounded-lg flex flex-col items-center justify-center">


                <div className="flex flex-col items-center ">
                    <h1 className="text-4xl font-bold">Welcome to Croptimizer 🌾</h1>
                    <p className="mt-4 text-lg text-center">
                        Croptimizer is a web application that helps farmers optimize their crop yield.
                    </p>

                    <div className="mt-8">
                        <Link to="/maps" className="px-4 py-2 bg-blue-500 text-white rounded-md">Get Started</Link>
                    </div>

                </div>


            </div>

        </div>
        
    );
    }

export default Landing;