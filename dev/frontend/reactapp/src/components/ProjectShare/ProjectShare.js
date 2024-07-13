import React, { useEffect, useState } from 'react';
import './ProjectShare.css';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProjectRequest from './ProjectRequest';
import ProjectTile from './ProjectTile';
import { getProjectInfo } from '../../db/firebaseFunction';
import ProjectShareHeader from './ProjectShareHeader';

function ProjectShare() {
  const [image, setImage] = useState(true);
  const [reinforcement, setReinforcement] = useState(false);
  const [imageProjects, setImageProjects] = useState(null);
  const [joinProject, setJoinProject] = useState(false)
  useEffect(() => {
    const fatchProjects = async () => {
      const projectsInfo = await getProjectInfo();
      setImageProjects(projectsInfo);
    };
    fatchProjects();
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
      {(image && imageProjects) ? (
        Object.keys(imageProjects).sort().map((key) => (
          <div key={key}>
            <ProjectTile title={key} info={imageProjects[key].toString()} style1={styleImage} />
          </div>
        ))
      ) : (<></>)}
    </div>
  )
}

export default ProjectShare;
