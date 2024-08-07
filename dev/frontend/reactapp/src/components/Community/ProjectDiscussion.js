import React, { useEffect, useState } from 'react'
import DiscussionTile from './DiscussionTile';
import './Community.css';
import { ReactComponent as EditIcon } from '../../assets/svg/edit.svg';
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg'
import { getDiscussionInfos } from '../../db/function/discussion';
import { useParams } from 'react-router-dom';

function ProjectDiscussion({ handleEdit, handleInfo, setReportInfo, setDiscussionId }) {
  const { projectName } = useParams()
  const [inputValue, setInputValue] = useState('');
  const [discussions, setDiscussions] = useState([]);
  useEffect(() => {
    const fetchDiscussion = async () => {
      const discussionInfo = await getDiscussionInfos(projectName);
      setDiscussions(discussionInfo);
    };

    fetchDiscussion();

  }, [projectName]);

  const handleClick = (reportInfo, discussionId) => {
    handleInfo();
    setDiscussionId(discussionId)
    setReportInfo(reportInfo);
  }
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
          <div className='post-button' onClick={handleEdit}>
            <p>投稿する</p>
            <EditIcon className='edit-icon' />
          </div>
        </div>
      </div>
      <div className='discussion-field'>
        {discussions ? (
          discussions.map((discussion) => (
            <div onClick={() => handleClick(discussion.data(), discussion.id)} key={discussion.id}>
              <DiscussionTile title={discussion.data().title} />
            </div>
            // console.log(discussion.id)
          ))
        ) : (
          <></>
        )}
        {/* <div onClick={handleInfo}>
          <DiscussionTile />
        </div>
        <DiscussionTile />
        <DiscussionTile />
        <DiscussionTile />
        <DiscussionTile /> */}
      </div>
    </div>
  )
}

export default ProjectDiscussion;
