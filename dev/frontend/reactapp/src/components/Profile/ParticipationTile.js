import React from 'react'
import './Profile.css'

function ParticipationTile({ projectName, rank }) {
  return (
    <div className='participation-wrapper'>
      <div className='participation-name'>
        <p>{projectName}</p>
      </div>
      <div className='participation-rank'>
        <p>{String(rank)}</p>
      </div>
    </div>
  )
}

export default ParticipationTile;
