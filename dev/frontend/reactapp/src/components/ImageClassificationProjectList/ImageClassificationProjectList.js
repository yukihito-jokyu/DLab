import React from 'react'
import ImageProjectField from './ImageProjectField'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import './ImageClassificationProjectList.css'
import CreateBackground from './CreateBackground'
import ContentsBackground from '../../uiParts/component/ContentsBackground'

function ImageClassificationProjectList() {
  return (
    <div className='projectlist-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
      />
      <ContentsBackground children={ImageProjectField} />
      {/* <div className='create-background-field'>
        <CreateBackground />
      </div> */}
    </div>
  )
}

export default ImageClassificationProjectList;
