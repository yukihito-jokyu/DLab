import React, { useState } from 'react';
import './Community.css';

function ProjectHeader() {
  const [overview, setOverview] = useState(true);
  const [preview, setPreview] = useState(false);
  const [discussion, setDiscussion] = useState(false);
  const [readerBoard, setReaderBoard] = useState(false);

  const handleOverview = () => {
    setOverview(true);
    setPreview(false);
    setDiscussion(false);
    setReaderBoard(false);
  };
  const handlePreview = () => {
    setOverview(false);
    setPreview(true);
    setDiscussion(false);
    setReaderBoard(false);
  };
  const handleDiscussion = () => {
    setOverview(false);
    setPreview(false);
    setDiscussion(true);
    setReaderBoard(false);
  };
  const handleReaderBoard = () => {
    setOverview(false);
    setPreview(false);
    setDiscussion(false);
    setReaderBoard(true);
  };
  const style1 = {
    height: overview ? '54px' : '50px',
    backgroundColor: overview ? 'white' : '#D9D9D9',
    borderTop: overview ? 'solid 4px #17A277' : 'none',
    borderLeft: overview ? 'solid 4px #17A277' : 'none',
    borderRight: overview ? 'solid 4px #17A277' : 'none',
    color: overview ? '#17A277' : '#868686'
  }
  const style2 = {
    height: preview ? '54px' : '50px',
    backgroundColor: preview ? 'white' : '#D9D9D9',
    borderTop: preview ? 'solid 4px #17A277' : 'none',
    borderLeft: preview ? 'solid 4px #17A277' : 'none',
    borderRight: preview ? 'solid 4px #17A277' : 'none',
    color: preview ? '#17A277' : '#868686'
  }
  const style3 = {
    height: discussion ? '54px' : '50px',
    backgroundColor: discussion ? 'white' : '#D9D9D9',
    borderTop: discussion ? 'solid 4px #17A277' : 'none',
    borderLeft: discussion ? 'solid 4px #17A277' : 'none',
    borderRight: discussion ? 'solid 4px #17A277' : 'none',
    color: discussion ? '#17A277' : '#868686'
  }
  const style4 = {
    height: readerBoard ? '54px' : '50px',
    backgroundColor: readerBoard ? 'white' : '#D9D9D9',
    borderTop: readerBoard ? 'solid 4px #17A277' : 'none',
    borderLeft: readerBoard ? 'solid 4px #17A277' : 'none',
    borderRight: readerBoard ? 'solid 4px #17A277' : 'none',
    color: readerBoard ? '#17A277' : '#868686'
  }
  
  return (
    <div className='project-header'>
      <div className='project-header-wrapper'>
        <div className='project-header-item' onClick={handleOverview} style={style1}>
          <p>Overview</p>
        </div>
        <div className='project-header-item' onClick={handlePreview} style={style2}>
          <p>Preview</p>
        </div>
        <div className='project-header-item' onClick={handleDiscussion} style={style3}>
          <p>Discussion</p>
        </div>
        <div className='project-header-item' onClick={handleReaderBoard} style={style4}>
          <p>ReaderBoard</p>
        </div>
      </div>
      <div className='project-header-bottom'></div>
    </div>
  )
}

export default ProjectHeader
