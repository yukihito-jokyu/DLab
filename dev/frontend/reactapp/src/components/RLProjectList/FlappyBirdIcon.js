import React from 'react';
import FlappyBirdImage from '../../assets/images/project_image/FlappyBird/FlappyBird_image2.png';

function FlappyBirdIcon() {
  return (
    <div className='rl-project-icon-wrapper'>
      <div className='project-title-wrapper'>
        <div className='project-title'>
          <p>FlappyBird</p>
        </div>
      </div>
      <div className='project-image-wrapper'>
        <div className='project-image'>
          <img src={FlappyBirdImage} alt='' />
        </div>
      </div>
      <div className='project-info-wrapper'>
        <div className='project-info'>
          <p>Flappy Birdは、鳥（Flappy Bird）がパイプの間をぶつからないように飛び続けることが目的です。<br/>鳥がパイプにぶつかったり、地面に落ちたりするとゲームオーバーになります。</p>
        </div>
      </div>
    </div>
  )
}

export default FlappyBirdIcon
