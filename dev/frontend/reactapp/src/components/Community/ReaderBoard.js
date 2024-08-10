import React, { useEffect, useState } from 'react';
import './Community.css';
import { useNavigate } from 'react-router-dom';

function ReaderBoard({ rank, name, accuracy, tileUserId }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [tileColor, setTileColor] = useState();
  const navigate = useNavigate();
  useEffect(() => {
    const initColor = () => {
      if (userId === tileUserId) {
        if (rank === 1) {
          const color = {
            backgroundColor: '#ffd448df',
            border: 'solid 5px #D997FF'
          }
          setTileColor(color)
        } else if (rank === 2) {
          const color = {
            backgroundColor: '#E9E9E9',
            border: 'solid 5px #D997FF'
          }
          setTileColor(color)
        } else if (rank === 3) {
          const color = {
            backgroundColor: '#F0BA7C',
            border: 'solid 5px #D997FF'
          }
          setTileColor(color)
        } else {
          const color = {
            border: 'solid 5px #D997FF'
          }
          setTileColor(color)
        }
      } else {
        if (rank === 1) {
          const color = {
            backgroundColor: '#ffd448df'
          }
          setTileColor(color)
        } else if (rank === 2) {
          const color = {
            backgroundColor: '#E9E9E9'
          }
          setTileColor(color)
        } else if (rank === 3) {
          const color = {
            backgroundColor: '#F0BA7C'
          }
          setTileColor(color)
        }
      }
      
    };
    initColor()
  }, [rank, userId, tileUserId]);
  const handleNav = () => {
    navigate(`/profile/${tileUserId}`);
  }
  return (
    <div className='reader-board-wrapper' style={tileColor} onClick={handleNav}>
      <div className='reader-board-rank'>
        {rank === 1 ? (
          <div>
            <p>{rank + 'st'}</p>
          </div>
        ) : rank === 2 ? (
          <div>
            <p>{rank + 'nd'}</p>
          </div>
        ) : rank === 3 ? (
          <div>
            <p>{rank + 'rd'}</p>
          </div>
        ) : (
          <div>
            <p>{rank}</p>
          </div>
        )}
      </div>
      <div className='reader-board-name'>
        <p>{name}</p>
      </div>
      <div className='reader-board-eval'>
        <p>{accuracy}</p>
      </div>
    </div>
  )
}

export default ReaderBoard
