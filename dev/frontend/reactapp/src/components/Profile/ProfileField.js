import React from 'react'
import ParticipationTile from './ParticipationTile';

function profileField() {
  return (
    <div className='profile-field-wrapper'>
      <div className='profile-title'>
        <p>Profile</p>
      </div>
      <div className='profile'>
        <div className='profile-left'>
          <div className='picture-field'></div>
        </div>
        <div className='profile-right'>
          <div className='profile-info'>
            <p className='user-name'>User Name</p>
            <p className='registration-date'>登録日：2024/06/08</p>
          </div>
        </div>
      </div>
      <div className='profile-project-list-wrapper'>
        <div className='project-list-header'>
          <div className='project-header'>
            <p>Projects</p>
          </div>
          <div className='rank-header'>
            <p>Rank</p>
          </div>
        </div>
        <div className='project-list-field'>
          <ParticipationTile />
          <ParticipationTile />
          <ParticipationTile />
          <ParticipationTile />
          <ParticipationTile />
        </div>
      </div>
    </div>
  )
}

export default profileField;
