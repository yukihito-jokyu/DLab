import React from 'react'
import { ReactComponent as EditIcon } from '../../assets/svg/edit.svg';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_40.svg';
import './Community.css';

function Discussion({ handleEdit }) {
  return (
    <div className='wrapper-out'>
      <div className='discussion-wrapper'>
        <div className='discussion-title-wrapper'>
          <div className='title-box'>
            <p>記事の新規作成</p>
          </div>
          <div className='delet-box' onClick={handleEdit}>
            <DeletIcon className='delet-icon' />
          </div>
        </div>
        <div className='title-input-wrapper'>
          <input
            type='text'
            placeholder='タイトルを入力'
          />
        </div>
        <div className='contents-wrapper'>
          <textarea
            type='text'
            placeholder='内容を入力'
          />
        </div>
        <div className='post-button-wrapper'>
          <div className='post-button'>
            <p>投稿する</p>
            <EditIcon className='edit-icon' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Discussion;
