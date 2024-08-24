import React from 'react';
import './Community.css';
import { useParams } from 'react-router-dom';

function ProjectHeader({ handleOverview, handlePreview, handleDiscussion, handleReaderBoard, overview, preview, discussion, readerBoard }) {
  const { task } = useParams();

  const baseClass = 'project-header-item';
  const dynamicClass = task === 'ReinforcementLearning' ? ' reinforcement' : '';

  const style1 = {
    height: overview ? '54px' : '50px',
    backgroundColor: overview ? 'white' : '#D9D9D9',
    borderTop: overview ? 'solid 4px #17A277' : 'none',
    borderLeft: overview ? 'solid 4px #17A277' : 'none',
    borderRight: overview ? 'solid 4px #17A277' : 'none',
    color: overview ? '#17A277' : '#868686',
    cursor: 'pointer'
  };

  const style2 = {
    height: preview ? '54px' : '50px',
    backgroundColor: preview ? 'white' : '#D9D9D9',
    borderTop: preview ? 'solid 4px #17A277' : 'none',
    borderLeft: preview ? 'solid 4px #17A277' : 'none',
    borderRight: preview ? 'solid 4px #17A277' : 'none',
    color: preview ? '#17A277' : '#868686',
    cursor: 'pointer'
  };

  const style3 = {
    height: discussion ? '54px' : '50px',
    backgroundColor: discussion ? 'white' : '#D9D9D9',
    borderTop: discussion ? 'solid 4px #17A277' : 'none',
    borderLeft: discussion ? 'solid 4px #17A277' : 'none',
    borderRight: discussion ? 'solid 4px #17A277' : 'none',
    color: discussion ? '#17A277' : '#868686',
    cursor: 'pointer'
  };

  const style4 = {
    height: readerBoard ? '54px' : '50px',
    backgroundColor: readerBoard ? 'white' : '#D9D9D9',
    borderTop: readerBoard ? 'solid 4px #17A277' : 'none',
    borderLeft: readerBoard ? 'solid 4px #17A277' : 'none',
    borderRight: readerBoard ? 'solid 4px #17A277' : 'none',
    color: readerBoard ? '#17A277' : '#868686',
    cursor: 'pointer'
  };

  return (
    <div className='project-header'>
      <div className='project-header-wrapper'>
        <div className={baseClass + dynamicClass} onClick={handleOverview} style={style1}>
          <p>Overview</p>
        </div>

        {task !== 'ReinforcementLearning' && (
          <div className={baseClass + dynamicClass} onClick={handlePreview} style={style2}>
            <p>Preview</p>
          </div>
        )}

        <div className={baseClass + dynamicClass} onClick={handleDiscussion} style={style3}>
          <p>Discussion</p>
        </div>
        <div className={baseClass + dynamicClass} onClick={handleReaderBoard} style={style4}>
          <p>LeaderBoard</p>
        </div>
      </div>
      <div className='project-header-bottom'></div>
    </div>
  );
}

export default ProjectHeader;
