import React from 'react'
import './Top.css';
import Explanation from './Explanation';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import Recommend from './Recommend';
import ProjectRecommend from './ProjectRecommend';
import YoutubeRecommend from './YoutubeRecommend';
import UpDatesRecommend from './UpDatesRecommend';
import Footer from './Footer';
import UserNameModal from './UserNameModal';


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
      <YoutubeRecommend />
      <UpDatesRecommend />
      <Footer />
      <UserNameModal />
    </div>
  )
}

export default Top
