import React from 'react';
import './Community.css';

function ProjectOverview({ summary, dataFormat, source }) {
  return (
    <div className='project-overview-wrapper'>
      <div className='overview-wrapper'>
        <p className='overview-title'>1. Summary</p>
        <p className='overview-info' dangerouslySetInnerHTML={{ __html: summary }}></p>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>2. Data format</p>
        <p className='overview-info' dangerouslySetInnerHTML={{ __html: dataFormat }}></p>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>3. Source</p>
        <p className='overview-info' dangerouslySetInnerHTML={{ __html: source }}></p>
      </div>
    </div>
  );
};

export default ProjectOverview;
