import React, { useState } from 'react'
import { ReactComponent as EditIcon } from '../../assets/svg/edit.svg';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_40.svg';
import './Community.css';
import { getUserName } from '../../db/function/users';
import { postArticle } from '../../db/function/discussion';
import { useParams } from 'react-router-dom';

function Discussion({ handleEdit }) {
  const { projectName } = useParams()
  const [title, setTitle] = useState('');
  const [comment, setComment] = useState('');
  const handlePost = async () => {
    const userId = JSON.parse(sessionStorage.getItem('userId'));
    const userName = await getUserName(userId);
    await postArticle(projectName, userId, userName, title, comment);
    handleEdit();
  };
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
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>
        <div className='contents-wrapper'>
          <textarea
            type='text'
            placeholder='内容を入力'
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </div>
        <div className='post-button-wrapper'>
          <div className='post-button' onClick={handlePost}>
            <p>投稿する</p>
            <EditIcon className='edit-icon' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Discussion;
