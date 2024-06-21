import React from 'react';
import './Community.css';

function ReaderBoard({ rank, name, accuracy }) {
  return (
    <div className='reader-board-wrapper'>
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
