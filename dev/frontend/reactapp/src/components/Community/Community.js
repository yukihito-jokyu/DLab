import React, { useEffect, useState } from 'react';
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
import ProjectReaderBoard from './ProjectReaderBoard';
import { getProjectInfoUp } from '../../db/firebaseFunction';

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

  const [projectName, setProjectName] = useState('');
  const [shortExp, setShortExp] = useState('');
  const [source, setSource] = useState('');
  const [sourceLink, setSourceLink] = useState('');
  const [summary, setSummary] = useState('');
  const [labelList, setLabelList] = useState([])

  // ページに訪れた時にproject_infoから情報を抜き取る
  useEffect(() => {
    const fatchProjects = async () => {
      const projectId = JSON.parse(sessionStorage.getItem('projectId'));
      const projectsInfo = await getProjectInfoUp(projectId);
      if (projectsInfo) {
        setProjectName(projectsInfo.name);
        setShortExp(projectsInfo.short_explanation);
        setSource(projectsInfo.source);
        setSourceLink(projectsInfo.source_link);
        setSummary(projectsInfo.summary);
        setLabelList(projectsInfo.label_list);
      }
    }

    fatchProjects()
    
  }, [])

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
      <ProjectActivate projectName={projectName} shortExp={shortExp} />
      <ProjectHeader {...props} />
      {overview && <ProjectOverview summary={summary} source={source} sourceLink={sourceLink} />}
      {preview && <ProjectPreview labelList={labelList} />}
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
        {readerBoard && <ProjectReaderBoard />}
    </div>
  );
};

export default Community;
