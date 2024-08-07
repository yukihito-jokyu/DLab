import React, { useContext } from 'react';
import '../css/ProjectModelName.css';
import { UserInfoContext } from '../../App';
import { useParams } from 'react-router-dom';

function ProjectModelName() {
  const { projectName } = useParams()
  return (
    <div className='project-model-name-wrapper'>
      <p>{projectName} - モデル選択</p>
    </div>
  )
}

export default ProjectModelName
