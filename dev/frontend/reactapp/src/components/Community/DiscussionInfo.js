import React, { useEffect, useState } from 'react';
import './Community.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_40.svg';
import { ReactComponent as SendIcon } from '../../assets/svg/send.svg';
import DiscussionComment from './DiscussionComment';
import { addDiscussionComment, getDiscussionComment, getDiscussionTitle } from '../../db/function/discussion';
import { getUserName } from '../../db/function/users';
import { useParams } from 'react-router-dom';

function DiscussionInfo({ handleInfo, discussionId }) {
  const { projectName } = useParams()
  const [comment, setComment] = useState('');
  const [title, setTitle] = useState('');
  const [reportInfo, setReportInfo] = useState([])
  const [push, setPush] = useState(false);
  useEffect(() => {
    const getDiscussion = async () => {
      const reportInfo = await getDiscussionComment(projectName, discussionId);
      const title = await getDiscussionTitle(projectName, discussionId);
      setTitle(title)
      setReportInfo(reportInfo);
    };

    getDiscussion();

  }, [discussionId, push, projectName]);
  const handlePosetComment = async () => {
    const userId = JSON.parse(sessionStorage.getItem('userId'));
    const userName = await getUserName(userId);
    await addDiscussionComment(projectName, discussionId, comment, userId, userName);
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
