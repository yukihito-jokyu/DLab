import React, { useEffect, useState } from 'react';
import './ProjectShare.css';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProjectRequest from './ProjectRequest';
import ProjectTile from './ProjectTile';
import ProjectShareHeader from './ProjectShareHeader';
import { getClassificationProjectInfo, getReinforcementlearningProjectInfo } from '../../db/function/project_info';
import { getJoinProject } from '../../db/function/users';

function ProjectShare() {
  const [image, setImage] = useState(true);
  const [rl, setRl] = useState(null);
  const [reinforcement, setReinforcement] = useState(false);
  const [imageProjects, setImageProjects] = useState(null);
  const [joinProject, setJoinProject] = useState(false);
  const [joinProjectList, setJoinProjectList] = useState(null);
  

  // 画像分類プロジェクト情報と参加プロジェクトの取得
  useEffect(() => {
    const fatchProjects = async () => {
      const userId = JSON.parse(sessionStorage.getItem('userId'));
      const projectsInfo = await getClassificationProjectInfo();
      const joinProject = await getJoinProject(userId);
      setImageProjects(projectsInfo);
      setJoinProjectList(joinProject)
    };
    fatchProjects();
  }, []);

  // 強化学習プロジェクト情報の取得
  useEffect(() => {
    const fatchRlProjects = async () => {
      const rlProjectInfo = await getReinforcementlearningProjectInfo();
      setRl(rlProjectInfo);
    };
    fatchRlProjects();
  }, []);
  const handleClickImage = () => {
    setImage(true);
    setReinforcement(false);
    setJoinProject(false);
  };
  const handleClickReinforcement = () => {
    setImage(false);
    setReinforcement(true);
    setJoinProject(false);
  };
  const handleJoin = () => {
    setJoinProject(!joinProject);
  }
  const styleImage = {
    background: "linear-gradient(90deg, #00C2FF 0%, #509FE8 100%)"
  }
  const styleRl = {
    background: "linear-gradient(90deg, #ff7aab 0%, rgba(156, 75, 105, 0.1) 100%)"
  }
  return (
    <div className='project-share-wrapper'>
      <div className='header-wrapper'>
        <Header
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <ProjectRequest />
      <ProjectShareHeader
        handleClickImage={handleClickImage}
        handleClickReinforcement={handleClickReinforcement}
        image={image}
        reinforcement={reinforcement}
        joinProject={joinProject}
        handleJoin={handleJoin}
      />
      {(image && imageProjects && !joinProject) ? (
        Object.keys(imageProjects).sort().map((key) => (
          <div key={key}>
            <ProjectTile title={key} info={imageProjects[key].toString()} style1={styleImage} />
          </div>
        ))
      ) : (image && imageProjects && joinProject) ? (
        joinProjectList.sort().map((value, index) => (
          // console.log(key)
          <div key={index}>
            <ProjectTile title={value.project_name} info={imageProjects[value.project_name].toString()} style1={styleImage} />
          </div>
        ))
      ) : (<></>)}
      {(!image && rl) ? (
        Object.keys(rl).sort().map((key) => (
          <div key={key}>
            <ProjectTile title={key} info={rl[key].toString()} style2={styleRl} />
          </div>
        ))
      ) : (<></>)}
    </div>
  )
}

export default ProjectShare;
