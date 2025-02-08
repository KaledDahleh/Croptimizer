import { Link } from "react-router-dom";

export const NavBar = () => {
    return (
        <div className="bg-green-500 p-4 flex justify-between items-center h-20">
            <div className="text-white text-2xl font-bold">Croptimizer</div>
            <div>
                <Link to="/maps" className="text-white mr-4">Maps</Link>
                <Link to="/about" className="text-white">About</Link>
            </div>
            
        </div>
    );
}