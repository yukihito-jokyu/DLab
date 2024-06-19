import React, { useState } from 'react';
import './Community.css';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProjectActivate from './ProjectActivate';
import ProjectHeader from './ProjectHeader';
import ProjectOverview from './ProjectOverview';
import ProjectPreview from './ProjectPreview';
import ProjectDiscussion from './ProjectDiscussion';
import Discussion from './Discussion';
import DiscussionInfo from './DiscussionInfo';

function Community() {
  const [overview, setOverview] = useState(true);
  const [preview, setPreview] = useState(false);
  const [discussion, setDiscussion] = useState(false);
  const [readerBoard, setReaderBoard] = useState(false);
  const [discussionEdit, setDiscussionEdit] = useState(false);
  const [discussionInfo, setDiscussionInfo] = useState(false)

  const handleOverview = () => {
    setOverview(true);
    setPreview(false);
    setDiscussion(false);
    setReaderBoard(false);
  };
  const handlePreview = () => {
    setOverview(false);
    setPreview(true);
    setDiscussion(false);
    setReaderBoard(false);
  };
  const handleDiscussion = () => {
    setOverview(false);
    setPreview(false);
    setDiscussion(true);
    setReaderBoard(false);
  };
  const handleReaderBoard = () => {
    setOverview(false);
    setPreview(false);
    setDiscussion(false);
    setReaderBoard(true);
  };
  const props = {
    handleOverview,
    handlePreview,
    handleDiscussion,
    handleReaderBoard,
    overview,
    preview,
    discussion,
    readerBoard
  };

  const handleEdit = () => {
    setDiscussionEdit(!discussionEdit);
  }
  const handleInfo = () => {
    setDiscussionInfo(!discussionInfo);
  }
  return (
    <div className='community-wrapper'>
      <div className='community-header-wrapper'>
        <Header 
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <ProjectActivate />
      <ProjectHeader {...props} />
      {overview && <ProjectOverview />}
      {preview && <ProjectPreview />}
      {discussion ? (
        discussionEdit ? (
          <Discussion handleEdit={handleEdit} />
        ) : discussionInfo ? (
          <DiscussionInfo handleInfo={handleInfo} />
        ) : (
          <ProjectDiscussion handleEdit={handleEdit} handleInfo={handleInfo} />
        )) : (
          <></>
        )}
    </div>
  );
};

export default Community;
