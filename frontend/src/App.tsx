import { BrowserRouter } from 'react-router-dom';
import Landing from './pages/Landing'
import './App.css'
import { NavBar } from './components/NavBar';
function App() {

  return (
    <BrowserRouter>

      <NavBar />
      <Landing />

    </BrowserRouter>
  )
}

export default App
