import React from 'react'
import ImageProjectField from '../../uniqueParts/component/ImageProjectField'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import '../css/ImageClassificationProjectList.css'
import Logo from '../../uiParts/component/Logo'
import CreateBackground from '../../uniqueParts/component/CreateBackground'

function ImageClassificationProjectList() {
  return (
    <div className='projectlist-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
      />
      <ImageProjectField />
      {/* <div className='test'></div> */}
      <div className='create-background-field'>
        <CreateBackground />
      </div>
    </div>
  )
}

export default ImageClassificationProjectList;
