import React, { useEffect, useState } from 'react';
import './Community.css';
import { ReactComponent as ChatSVG } from '../../assets/svg/chat_bubble_24.svg' 
import { getImage } from '../../db/function/storage';

function DiscussionTile({ title, discussionUserId, commentNum }) {
  const [userImage, setUserImage] = useState();
  useEffect(() => {
    const fetchUserIcon = async () => {
      console.log(discussionUserId)
      const path = `images/${discussionUserId}`;
      const userIcon = await getImage(path)
      setUserImage(userIcon);
    };
    fetchUserIcon();
  }, [discussionUserId]);
  return (
    <div className='discussion-tile-wrapper'>
      <div className='user-icon'>
        <img src={userImage} alt='user-icon' />
      </div>
      <p>{title}</p>
      <div className='comment-wrapper'>
        <div className='comment-icon'>
          <ChatSVG className='chat-svg' />
        </div>
        <div className='comment-num-wrapper'>
          <p>{commentNum}</p>
        </div>
      </div>
    </div>
  )
}

export default DiscussionTile;
