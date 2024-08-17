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
  const [filteredDiscussions, setFilteredDiscussions] = useState([]);

  useEffect(() => {
    const fetchDiscussion = async () => {
      const discussionInfo = await getDiscussionInfos(projectName);
      setDiscussions(discussionInfo);
      setFilteredDiscussions(discussionInfo);
    };

    fetchDiscussion();

  }, [projectName]);

  const handleClick = (reportInfo, discussionId) => {
    handleInfo();
    setDiscussionId(discussionId)
    setReportInfo(reportInfo);
  }

  const handleSearch = () => {
    const filtered = discussions.filter(discussion =>
      discussion.data().title.toLowerCase().includes(inputValue.toLowerCase())
    );
    setFilteredDiscussions(filtered);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className='project-discussion-wrapper'>
      <div className='discussion-search-wrapper'>
        <div className='discussion-search'>
          <div className='input-field'>
            <input
              type='text'
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder='フリーワード検索'
            />
            <div className='search-field' onClick={handleSearch} style={{ cursor: 'pointer' }}>
              <SearchIcon />
            </div>
          </div>
          <div className='post-button' onClick={handleEdit} style={{ cursor: 'pointer' }}>
            <p>投稿する</p>
            <EditIcon className='edit-icon' />
          </div>
        </div>
      </div>
      <div className='discussion-field'>
        {filteredDiscussions.length > 0 ? (
          filteredDiscussions.map((discussion) => (
            <div onClick={() => handleClick(discussion.data(), discussion.id)} key={discussion.id} style={{ cursor: 'pointer' }}>
              <DiscussionTile
                title={discussion.data().title}
                discussionUserId={discussion.data().user_id}
                commentNum={discussion.data().comments.length}
              />
            </div>
          ))
        ) : (
          <p className='none-text'>該当するディスカッションが見つかりませんでした。</p>
        )}
      </div>
    </div>
  )
}

export default ProjectDiscussion;
