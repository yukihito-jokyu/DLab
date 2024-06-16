import React from 'react';
import './ImageClassificationProjectList.css';
import { ReactComponent as AddIcon } from '../../assets/svg/project_add_48.svg'
import { useNavigate } from 'react-router-dom';

function ImageProjectAdd() {
  const navigate = useNavigate();
  const handleNav = () => {
    navigate('/projectshare');
  }
  return (
    <div className='ImageProjectAdd-wrapper' onClick={handleNav}>
      <AddIcon className='add-svg' />
    </div>
  )
};

export default ImageProjectAdd;
