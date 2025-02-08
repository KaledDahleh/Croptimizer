import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing'
import Maps from './pages/Maps'
import './App.css'
import { NavBar } from './components/NavBar';
import { Form } from './pages/Form';

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>

        <Route path="/" element={<Landing />} />
        <Route path="/maps" element={<Maps />} />
        <Route path="/form" element={<Form />} />
        <Route path="/about"/>

      </Routes>
    </BrowserRouter>
  );
}

export default App
