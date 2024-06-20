import React from 'react';
import './Community.css';

function ProjectOverview({ summary, source, sourceLink }) {
  return (
    <div className='project-overview-wrapper'>
      <div className='overview-wrapper'>
        <p className='overview-title'>1. Summary</p>
        <p className='overview-info'>{summary}</p>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>2. Data format</p>
        <p className='overview-info'>データの形式は、以下の通りです。</p>
        <div></div>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>3. Source</p>
        <p className='overview-info'>{source}</p>
        
      </div>
    </div>
  );
};

export default ProjectOverview
