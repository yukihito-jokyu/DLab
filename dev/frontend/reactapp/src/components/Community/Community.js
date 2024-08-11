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
import AlertModal from '../utils/AlertModal';
import { useNavigate, useParams } from 'react-router-dom';
import { getProjectDetailedInfo } from '../../db/function/project_info';
import { updateJoinProject } from '../../db/function/users';
import UserIcon from '../../uiParts/component/UserIcon';

function Community() {
  const { projectName } = useParams();
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [overview, setOverview] = useState(true);
  const [preview, setPreview] = useState(false);
  const [discussion, setDiscussion] = useState(false);
  const [readerBoard, setReaderBoard] = useState(false);
  const [discussionEdit, setDiscussionEdit] = useState(false);
  const [discussionInfo, setDiscussionInfo] = useState(false);
  const [joinConfirmationModal, setJoinConfirmationModal] = useState(false);

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

  // const [projectName, setProjectName] = useState('');
  const [shortExp, setShortExp] = useState('');
  const [source, setSource] = useState('');
  const [sourceLink, setSourceLink] = useState('');
  const [summary, setSummary] = useState('');
  const [labelList, setLabelList] = useState([]);

  // 参加モーダル表示・非表示
  const [joinModal, setJoinModal] = useState(false);
  // モーダル表示・非表示関数
  const changeJoinModal = () => {
    setJoinModal(!joinModal);
  };

  const navigate = useNavigate();
  // プロジェクト参加関数
  const handleJoin = async () => {
    await updateJoinProject(userId, projectName);
    const sentData = {
      "user_id": userId,
      "project_name": projectName
    }
    const response = await fetch('http://127.0.0.1:5000/mkdir/project', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sentData),
    });
    console.log(response);
    setJoinConfirmationModal(true);
    
  };
  const handleNav = () => {
    setJoinConfirmationModal(false);
    navigate('/ImageClassificationProjectList');
  }


  // ページに訪れた時にproject_infoから情報を抜き取る
  useEffect(() => {
    const fatchProjects = async () => {
      const projectsInfo = await getProjectDetailedInfo(projectName);
      if (projectsInfo) {
        // setProjectName(projectsInfo.name);
        setShortExp(projectsInfo.short_explanation);
        setSource(projectsInfo.source);
        setSourceLink(projectsInfo.source_link);
        setSummary(projectsInfo.summary);
        setLabelList(projectsInfo.label_list);
      }
    };
    fatchProjects();
  }, [projectName]);

  // 記事のid
  const [reportInfo, setReportInfo] = useState('');
  const [discussionId, setDiscussionId] = useState('');
  

  const handleEdit = () => {
    setDiscussionEdit(!discussionEdit);
  }
  const handleInfo = () => {
    setDiscussionInfo(!discussionInfo);
  }
  const sendText = 'プロジェクトをアクティベートします。<br/>よろしいですか？'
  return (
    <div className='community-wrapper'>
      <div className='community-header-wrapper'>
        <Header 
          burgerbutton={BurgerButton}
          logocomponent={Logo}
          usericoncomponent={UserIcon}
        />
      </div>
      <ProjectActivate projectName={projectName} shortExp={shortExp} changeJoinModal={changeJoinModal} />
      <ProjectHeader {...props} />
      {overview && <ProjectOverview summary={summary} source={source} sourceLink={sourceLink} />}
      {preview && <ProjectPreview labelList={labelList} />}
      {discussion ? (
        discussionEdit ? (
          <Discussion handleEdit={handleEdit} />
        ) : discussionInfo ? (
          <DiscussionInfo handleInfo={handleInfo} discussionId={discussionId} />
        ) : (
          <ProjectDiscussion handleEdit={handleEdit} handleInfo={handleInfo} setReportInfo={setReportInfo} setDiscussionId={setDiscussionId} />
        )) : (
          <></>
        )}
        {readerBoard && <ProjectReaderBoard />}
        {joinModal && (<AlertModal deleteModal={changeJoinModal} handleClick={handleJoin} sendText={sendText} />)}
        {joinConfirmationModal && (<AlertModal deleteModal={handleNav} handleClick={handleNav} sendText={'プロジェクトに参加しました。'} />)}
    </div>
  );
};

export default Community;
