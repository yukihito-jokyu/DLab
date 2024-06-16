import React, { useState } from 'react'
import DiscussionTile from './DiscussionTile';
import './Community.css';
import { ReactComponent as EditIcon } from '../../assets/svg/edit.svg';
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg'

function ProjectDiscussion() {
  const [inputValue, setInputValue] = useState('')
  return (
    <div className='project-discussion-wrapper'>
      <div className='discussion-search-wrapper'>
        <div className='discussion-search'>
          <div className='input-field'>
            <input
              type='text'
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder='フリーワード検索'
            />
            <div className='search-field'>
              <SearchIcon />
            </div>
          </div>
          <div className='post-button'>
            <p>投稿する</p>
            <EditIcon className='edit-icon' />
          </div>
        </div>
      </div>
      <div className='discussion-field'>
        <DiscussionTile />
        <DiscussionTile />
        <DiscussionTile />
        <DiscussionTile />
        <DiscussionTile />
      </div>
    </div>
  )
}

export default ProjectDiscussion;
