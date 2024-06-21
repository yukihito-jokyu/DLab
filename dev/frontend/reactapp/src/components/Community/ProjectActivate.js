import React, { useEffect, useState } from 'react';
import './Community.css';
import { getProjectInfo } from '../../db/firebaseFunction';
import GradationButton from '../../uiParts/component/GradationButton';
import CIFA10Image from '../../assets/images/CIFA10Image.png';
import { useNavigate } from 'react-router-dom';

function ProjectActivate({ projectName, shortExp }) {
  const style1 = {
    width: '200px',
    background: 'linear-gradient(95.34deg, #B6F862 3.35%, #00957A 100%), linear-gradient(94.22deg, #D997FF 0.86%, #50BCFF 105.96%)'
  };
  const navigate = useNavigate();
  const handleNav = () => {
    navigate('/ImageClassificationProjectList');
  };
  return (
    <div className='project-activate-wrapper'>
      <div className='activate-left'>
        {projectName ? (
          <div className='activate-info-field'>
            <p className='activate-project-title'>{projectName}</p>
            <p dangerouslySetInnerHTML={{ __html: shortExp }} className='activate-project-info'></p>
          </div>
        ) : (<></>)}
        <div className='activate-button-wrapper' onClick={handleNav}>
          <GradationButton text={'Activate'} style1={style1} />
        </div>
      </div>
      <div className='activate-right'>
        <img src={CIFA10Image} alt='ProjectImage' className='activate-project-image' />
      </div>
    </div>
  );
};

export default ProjectActivate;
