import React from 'react'
import './Top.css';
import Explanation from './Explanation';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import Recommend from './Recommend';
import ProjectRecommend from './ProjectRecommend';


function Top() {
  return (
    <div className='top-wrapper'>
      <div className='top-header-wrapper'>
        <Header 
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <Explanation />
      <Recommend />
      <ProjectRecommend />
    </div>
  )
}

export default Top
