import React from 'react'
import { useNavigate } from 'react-router-dom'
import './Home.css';


function Home() {
  const navigate = useNavigate()
  const handleReinforcement = () => {
    navigate('/Reinforcement')
  }
  const handleImageRecognition = () => {
    navigate('/ImageRecognition')
  }

  return (
    <div className='home-page'>
      <h1>Home</h1>
      <div className='home-wrapper'>
        <div className='Reinforcement-button home-button-style' onClick={handleReinforcement}>
          <p>強化学習</p>
        </div>
        <div className='ImageRecognition-button home-button-style' onClick={handleImageRecognition}>
          <p>画像認識</p>
        </div>
      </div>
    </div>
  )
}

export default Home
