import React from 'react'
import { useNavigate } from 'react-router-dom'

// import CartPoleVideo from '../assets/movie/CartPoleRe.mp4'

import "./Reinforcement.css";

function Reinforcement() {
  

  const navigate = useNavigate();
  const handleCartpole = () => {
    navigate('/Reinforcement/Cartpole')
  };
  const handleFlappybird = () => {
    navigate('/Reinforcement/Flappybird')
  }
  return (
    <div className='Reinforcement-page'>
      <h1>Reinforcement</h1>
      <div className='Reinforcement-wrapper'>
        <div className='Reinforcement-button-style' onClick={handleCartpole}>Cartpole</div>
        <div className='Reinforcement-button-style' onClick={handleFlappybird}>Flappybird</div>
      </div>
    </div>
  )
}

export default Reinforcement
