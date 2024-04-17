import React from 'react'
import ImageProjectField from '../../uniqueParts/component/ImageProjectField'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import '../css/ImageClassificationProjectList.css'
import Logo from '../../uiParts/component/Logo'

function ImageClassificationProjectList() {
  return (
    <div className='projectlist-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
      />
      <ImageProjectField />
    </div>
  )
}

export default ImageClassificationProjectList;
