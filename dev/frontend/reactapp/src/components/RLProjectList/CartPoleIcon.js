import React from 'react'
import './RLProjectList.css'
import CartPoleImage from '../../assets/images/project_image/CartPole/CartPole_image2.png';

function CartPoleIcon() {
  return (
    <div className='rl-project-icon-wrapper'>
      <div className='project-title-wrapper'>
        <div className='project-title'>
          <p>CartPole</p>
        </div>
      </div>
      <div className='project-image-wrapper'>
        <div className='project-image'>
          <img src={CartPoleImage} alt='' />
        </div>
      </div>
      <div className='project-info-wrapper'>
        <div className='project-info'>
          <p>CartPoleは、ポール（棒）がカート（台車）の上に垂直に立つようにバランスを取ることを目指します。<br/>カートを左右に動かして、ポールが倒れないようにバランスを取り続けることが目的です。</p>
        </div>
      </div>
    </div>
  )
}

export default CartPoleIcon;
