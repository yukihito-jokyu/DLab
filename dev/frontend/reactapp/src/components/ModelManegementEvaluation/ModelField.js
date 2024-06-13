import React, { useEffect, useState } from 'react';
import './ModelManegementEvaluation.css';
import ModelFieldHeader from './ModelFieldHeader';
import ModelTile from './ModelTile';
import ModelCreateButton from './ModelCreateButton';
import DLButton from './DLButton';
import { useNavigate } from 'react-router-dom';
import { getModelId } from '../../db/firebaseFunction';

function ModelField() {
  const [model, setModel] = useState(null)
  const porjectId = 'CartPole'
  useEffect(() => {
    const fatchProjects = async () => {
      const userId = JSON.parse(sessionStorage.getItem('userId'));
      const doc = await getModelId(userId, porjectId);
      setModel(doc);
      console.log(doc)
      // querySnapshot.docs.map((doc) => (
      //   console.log(doc)
      // ));
    };

    fatchProjects();

  }, [porjectId]);
  return (
    <div className='model-field-wrapper'>
      <ModelFieldHeader />
      <div className='tile-field'>
        {model ? (
          Object
        ) : (<></>)}
        <ModelTile />
        <ModelTile />
        <ModelCreateButton />
      </div>
      <div className='DL-field'>
        <DLButton />
      </div>
    </div>
  )
}

export default ModelField;
