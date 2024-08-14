import React from 'react';
import './Community.css';

function PreviewImage({ image }) {
  return (
    <div className='preview-image-wrapper'>
      <img src={`data:image/png;base64,${image}`} alt='test_image' />
    </div>
  )
}

export default PreviewImage
