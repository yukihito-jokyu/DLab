import React from 'react';
import './Community.css';
import { useParams } from 'react-router-dom';

function ProjectOverview({ summary, dataFormat, environment, source }) {
  const { task } = useParams();

  return (
    <div className='project-overview-wrapper'>
      <div className='overview-wrapper'>
        <p className='overview-title'>1. Summary</p>
        <p className='overview-info' dangerouslySetInnerHTML={{ __html: summary }}></p>
      </div>
      <div className='overview-wrapper'>
        {task === 'ImageClassification' ? (
          <>
            <p className='overview-title'>2. Data format</p>
            <p className='overview-info' dangerouslySetInnerHTML={{ __html: dataFormat }}></p>
          </>
        ) : task === 'ReinforcementLearning' ? (
          <>
            <p className='overview-title'>2. Environment</p>
            <p className='overview-info' dangerouslySetInnerHTML={{ __html: environment }}></p>
          </>
        ) : null}
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>3. Source</p>
        <p className='overview-info' dangerouslySetInnerHTML={{ __html: source }}></p>
      </div>
    </div>
  );
}

export default ProjectOverview;
