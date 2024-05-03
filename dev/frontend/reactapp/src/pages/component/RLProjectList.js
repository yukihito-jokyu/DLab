import React from 'react'
import '../css/RLProjectList.css'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import RLProjectField from '../../uniqueParts/component/RLProjectField'

function RLProjectList() { 
  return (
    <div className='rl-project-list-wrapper'>
      <Header 
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={null}
      />
      <RLProjectField />
    </div>
  )
}

export default RLProjectList