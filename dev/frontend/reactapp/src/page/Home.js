import React from 'react'
import { useNavigate } from 'react-router-dom'
import './Home.css';
import { socket } from '../socket/socket';


function Home() {
  const navigate = useNavigate()
  const handleReinforcement = () => {
    navigate('/Reinforcement')
  }
  const handleImageRecognition = () => {
    navigate('/ImageRecognition')
  }

  const handleClick = () => {
    const dataToSend = { message: 'Hello, server!' };

    try {
        socket.emit('test', dataToSend);
        socket.on('test_event', (response) => {
            console.log('Response from server:', response);
        });
    } catch (error) {
        // エラーメッセージを表示しないようにする
        console.error = () => {};
    }
  }

  const handleDir = async () => {
    const sentData = {
      test: 't'
    }
    const response = await fetch('http://127.0.0.1:5000/mkdir/test', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sentData),
    });
    console.log(response)
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
      <button onClick={handleClick}>ソケット</button>
      <button onClick={handleDir}>mkdir</button>
    </div>
  )
}

export default Home
