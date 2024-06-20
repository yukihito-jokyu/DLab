import React, { useEffect, useState } from 'react';
import './Community.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_40.svg';
import { ReactComponent as SendIcon } from '../../assets/svg/send.svg';
import DiscussionComment from './DiscussionComment';
import { addComment, getComment, getTitle, getUserName } from '../../db/firebaseFunction';

function DiscussionInfo({ handleInfo, discussionId }) {
  const [comment, setComment] = useState('');
  const [title, setTitle] = useState('');
  const [reportInfo, setReportInfo] = useState([])
  const [push, setPush] = useState(false);
  useEffect(() => {
    const getDiscussion = async () => {
      const projectId = JSON.parse(sessionStorage.getItem('projectId'));
      const reportInfo = await getComment(projectId, discussionId);
      const title = await getTitle(projectId, discussionId);
      setTitle(title)
      setReportInfo(reportInfo);
    };

    getDiscussion();

  }, [discussionId, push]);
  const handlePosetComment = async () => {
    const projectId = JSON.parse(sessionStorage.getItem('projectId'));
    const userId = JSON.parse(sessionStorage.getItem('userId'));
    const userName = await getUserName(userId);
    await addComment(projectId, discussionId, comment, userId, userName);
    setComment('');
    setPush(!push);
  }
  return (
    <div className='discussion-info-wrapper'>
      <div className='discussion-info-title-wrapper'>
        <div className='title-wrapper'>
          <p>{title}</p>
        </div>
        <div className='delet-box' onClick={handleInfo}>
          <DeletIcon className='delet-icon' />
        </div>
      </div>
      <div className='comment-field'>
        {reportInfo.length > 0 ? (
          reportInfo.map((comment, index) => (
            <div key={index}>
              <DiscussionComment comment={comment.comment} />
            </div>
            // console.log(comment)
          ))
        ) : (
          <></>
        )}
      </div>
      <div className='comment-push-field'>
        <input
          className='comment-push'
          type='text'
          placeholder='コメントを入力'
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        <div className='send-field'>
          <div className='send-middle-wrapper' onClick={handlePosetComment}>
            <p>コメント</p>
            <SendIcon className='send-icon' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default DiscussionInfo
