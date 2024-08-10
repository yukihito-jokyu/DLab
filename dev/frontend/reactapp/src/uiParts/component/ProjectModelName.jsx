import React, { useContext, useEffect, useState } from 'react';
import '../css/ProjectModelName.css';
import { UserInfoContext } from '../../App';
import { useParams } from 'react-router-dom';
import { getModelName } from '../../db/function/model_management';

function ProjectModelName() {
  const { projectName, modelId } = useParams()
  const [modelName, setModelName] = useState('');
  useEffect(() => {
    const fetchModelName = async () => {
      console.log(modelId)
      const name = await getModelName(modelId);
      setModelName(name);
    };
    if (modelId) {
      fetchModelName();
    }
  }, [modelId])
  return (
    <div className='project-model-name-wrapper'>
      {modelId ? (<p>{projectName} - {modelName}</p>) :(<p>{projectName} - {'モデル選択'}</p>)}
    </div>
  )
}

export default ProjectModelName
