import React from 'react'
import './RLProjectList.css'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import RLProjectField from './RLProjectField'
import UserIcon from '../../uiParts/component/UserIcon'

function RLProjectList() { 
  return (
    <div className='rl-project-list-wrapper'>
      <Header 
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        usericoncomponent={UserIcon}
        projectmodelnamecomponent={null}
      />
      <RLProjectField />
    </div>
  )
}

export default RLProjectList