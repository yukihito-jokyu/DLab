import React, { useContext } from 'react';
import '../css/ProjectModelName.css';
import { UserInfoContext } from '../../App';

function ProjectModelName() {
  // const { projectId } = useContext(UserInfoContext);
  const projectId = JSON.parse(sessionStorage.getItem('projectId'));
  return (
    <div className='project-model-name-wrapper'>
      <p>{projectId} - モデル選択</p>
    </div>
  )
}

export default ProjectModelName
