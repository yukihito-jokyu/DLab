import React, { useEffect, useState } from 'react';
import './ProjectShare.css';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProjectRequest from './ProjectRequest';
import ProjectTile from './ProjectTile';
import { getProjectInfo } from '../../db/firebaseFunction';

function ProjectShare() {
  const [projects, setProjects] = useState(null);
  useEffect(() => {
    const fatchProjects = async () => {
      const projectsInfo = await getProjectInfo();
      setProjects(projectsInfo);
    };

    fatchProjects();

  }, []);
  return (
    <div className='project-share-wrapper'>
      <div className='header-wrapper'>
        <Header
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <ProjectRequest />
      {projects ? (
        Object.keys(projects).sort().map((key) => (
          <div key={key}>
            <ProjectTile title={key} info={projects[key].toString()} />
          </div>
        ))
      ) : (<></>)}
    </div>
  )
}

export default ProjectShare;
