import React from 'react'
import { useNavigate } from 'react-router-dom'

// import CartPoleVideo from '../assets/movie/CartPoleRe.mp4'

import "./Reinforcement.css";

function Reinforcement(props) {
  // 引数取得
  const elementId = props.elementid;

  const navigate = useNavigate();
  const handleCartpole = (id) => {
    navigate('/Reinforcement/Cartpole'+id)
  };
  const handleFlappybird = () => {
    navigate('/Reinforcement/Flappybird')
  }
  // クリックしたときに要素追加のイベントが走る。
  const handleClickIvent = () => {
    props.handlemakeid()
  }
  return (
    <div className='Reinforcement-page'>
      <h1>Reinforcement</h1>
      <button onClick={handleClickIvent}>+</button>
      <div className='Reinforcement-wrapper'>
        {elementId.map((id, index) => (
          <div key={index} className='Reinforcement-button-style' onClick={() => handleCartpole(id)}>Cartpole</div>
        ))}
        <div className='Reinforcement-button-style' onClick={handleFlappybird}>Flappybird</div>
      </div>
    </div>
  )
}

export default Reinforcement
