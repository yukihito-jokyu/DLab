import React from 'react';
import './Community.css';

function DiscussionComment({ comment }) {
  return (
    <div className='discussion-comment-wrapper'>
      <div className='comment-left'>
        <div className='user-icon'></div>
        <div className='rod'></div>
      </div>
      <div className='comment-right'>
        <p>{comment}</p>
      </div>
    </div>
  )
}

export default DiscussionComment;