import React, { useEffect, useState } from 'react';
import GradationButton from '../../uiParts/component/GradationButton';

import ReactPlayer from 'react-player';

function YoutubeRecommend() {
  const [height, setHeight] = useState(0);
  useEffect(() => {
    const calculateHeight = () => {
      const element = document.getElementById('youtube-field-width');
      const width = element.offsetWidth / 3 * 2;
      setHeight(width);
    };

    calculateHeight()
    window.addEventListener('resize', calculateHeight);
    return () => window.removeEventListener('resize', calculateHeight);
  }, [])
  const youtubeStyle = {
    background: '#E37588'
  }
  const youtubeFieldStyle = {
    height: `${height}px`
  }
  return (
    <div className='youtube-wrapper'>
      <div className='youtube-title'>
        <p>WHAT IS THIS</p>
      </div>
      <div className='youtube-info-wrapper'>
        <div className='youtube-info' style={youtubeFieldStyle}>
          <div className='youtube-middle'>
            <p className='info-p'>下記リンクよりアプリケーションの<br />概要を知ることができます。</p>
            <div className='youtube-button'>
              <a href='https://youtu.be/7jI3Msmk9HI'>
                <GradationButton text={'Youtube'} style1={youtubeStyle} />
              </a>
            </div>
          </div>
        </div>
        <div className='youtube-field' id='youtube-field-width' style={youtubeFieldStyle}>
          <ReactPlayer
            url={"https://youtu.be/7jI3Msmk9HI"}
            width={"100%"}
            height={"100%"}
            controls={true}
          />
        </div>
      </div>
    </div>
  );
};

export default YoutubeRecommend;
