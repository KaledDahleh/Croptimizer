import { Link } from "react-router-dom";


export const NavBar = () => {
    return (
        <div className="bg-white p-4 flex justify-between items-center h-20">
            <Link to="/" className="text-2xl font-bold" style={{color: "#5C4033"}}>Croptimizer</Link>
            <div>
                <Link to="/maps" className="text-green-500 mr-4">Maps</Link>
                <Link to="/about" className="text-green-500">About</Link>
            </div>
            
        </div>
    );
}