import React, { useState } from 'react';
import './Community.css';
import PreviewImage from './PreviewImage';
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg'

function ProjectPreview({ labelList }) {
  const [selectedValue, setSelectedValue] = useState('');
  const [options, setOptions] = useState(["all", "bird", "fish", "dog"]);
  const handleSelectedChange = (e) => {
    setSelectedValue(e.target.values);
  };
  return (
    <div className='preview-wrapper'>
      <div className='preview-config-wrapper'>
        <div className='preview-config-field'>
          <p>数量</p>
          <input />
          <p>Label</p>
          <select value={selectedValue} onChange={handleSelectedChange}>
            {labelList.map((label, index) => (
              <option key={index} value={label}>
                {label}
              </option>
            ))}
          </select>
          <div className='search-button'>
            <SearchIcon className='search-icon' />
          </div>
        </div>
      </div>
      <div className='preview-image-field'>
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
        <PreviewImage />
      </div>
    </div>
  )
}

export default ProjectPreview
