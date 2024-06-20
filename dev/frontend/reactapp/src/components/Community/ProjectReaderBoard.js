import React from 'react'
import './Community.css'
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg';
import ReaderBoard from './ReaderBoard';

function ProjectReaderBoard() {
  return (
    <div className='project-reader-board-wrapper'>
      <div className='reader-board-setting'>
        <div className='my-rank-button'>
          <p>自分の順位</p>
        </div>
        <div className='favorite-button'>
          <p>お気に入り</p>
        </div>
        <div className='user-search'>
          <input
            type='text'
            placeholder='ユーザー検索'
          />
          <div className='search-wrapper'>
            <SearchIcon className='search-icon' />
          </div>
        </div>
      </div>
      <div className='reader-board-header'>
        <div className='reader-board-rank'>
          <p>Rank</p>
        </div>
        <div className='reader-board-name'>
          <p>User Name</p>
        </div>
        <div className='reader-board-eval'>
          <p>Accuracy</p>
        </div>
      </div>
      <div className='reader-board-field'>
        <ReaderBoard />
        <ReaderBoard />
        <ReaderBoard />
        <ReaderBoard />
        <ReaderBoard />
        <ReaderBoard />
      </div>
    </div>
  )
}

export default ProjectReaderBoard;
