import React from 'react';
import './Community.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_40.svg';
import { ReactComponent as SendIcon } from '../../assets/svg/send.svg';
import DiscussionComment from './DiscussionComment';

function DiscussionInfo({ handleInfo }) {
  return (
    <div className='discussion-info-wrapper'>
      <div className='discussion-info-title-wrapper'>
        <div className='title-wrapper'>
          <p>記事タイトル</p>
        </div>
        <div className='delet-box' onClick={handleInfo}>
          <DeletIcon className='delet-icon' />
        </div>
      </div>
      <div className='comment-field'>
        <DiscussionComment />
        <DiscussionComment />
        <DiscussionComment />
        <DiscussionComment />
        <DiscussionComment />
        <DiscussionComment />
      </div>
      <div className='comment-push-field'>
        <input
          className='comment-push'
          type='text'
          placeholder='コメントを入力'
        />
        <div className='send-field'>
          <div className='send-middle-wrapper'>
            <p>コメント</p>
            <SendIcon className='send-icon' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default DiscussionInfo
