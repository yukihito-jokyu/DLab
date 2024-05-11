import React from 'react'
import ImageProjectField from '../../uniqueParts/component/ImageProjectField'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import '../css/ImageClassificationProjectList.css'
import CreateBackground from '../../uniqueParts/component/CreateBackground'
import ContentsBackground from '../../uiParts/component/ContentsBackground'

function ImageClassificationProjectList() {
  return (
    <div className='projectlist-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
      />
      <ContentsBackground children={ImageProjectField} />
      <div className='create-background-field'>
        <CreateBackground />
      </div>
    </div>
  )
}

export default ImageClassificationProjectList;
