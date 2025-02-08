import React, { useState, useEffect } from 'react';

const Form: React.FC = () => {
    const [soilType, setSoilType] = useState('');
    const [crop, setCrop] = useState('');
    const [crops, setCrops] = useState<string[]>([]);

    useEffect(() => {
        // Fetch crop list from an API or define it here
        const fetchCrops = async () => {
            // Example static crop list
            const cropList = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Barley'];
            setCrops(cropList);
        };

        fetchCrops();
    }, []);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        console.log(`Soil Type: ${soilType}, Crop: ${crop}`);
    };

    return (
        <div className="form-container">
            <h1>Crop Optimizer</h1>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="soilType">Soil Type</label>
                    <select
                        id="soilType"
                        value={soilType}
                        onChange={(e) => setSoilType(e.target.value)}
                        required
                    >
                        <option value="">Select Soil Type</option>
                        <option value="clay">Clay</option>
                        <option value="loam">Loam</option>
                        <option value="sandy">Sandy</option>
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="crop">Crop</label>
                    <select
                        id="crop"
                        value={crop}
                        onChange={(e) => setCrop(e.target.value)}
                        required
                    >
                        <option value="">Select Crop</option>
                        {crops.map((crop, index) => (
                            <option key={index} value={crop}>
                                {crop}
                            </option>
                        ))}
                    </select>
                </div>
                <button type="submit">Submit</button>
            </form>
        </div>
    );
};

export default Form;