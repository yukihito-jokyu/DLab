import React, { useState } from 'react';
import './Community.css';
import PreviewImage from './PreviewImage';
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg'
import { useParams } from 'react-router-dom';

function ProjectPreview({ labelList }) {
  const { projectName } = useParams();
  const [selectedValue, setSelectedValue] = useState(labelList[0]);
  const [imageNum, setImageNum] = useState(1);
  const [images, setImages] = useState();
  const handleSelectedChange = (e) => {
    setSelectedValue(e.target.value);
  };

  const handleChange = (event) => {
    const inputValue = event.target.value;
    // 数値以外の入力や範囲外の入力を処理
    const numericValue = parseInt(inputValue, 10);
    if (!isNaN(numericValue)) {
      if (numericValue < 1) {
        setImageNum(1);
      } else if (numericValue > 100) {
        setImageNum(100);
      } else {
        setImageNum(numericValue);
      }
    } else {
      setImageNum('');
    }
  };

  // backendとの通信
  const handleGetImage = async () => {
    const sentData = {
      'dataset': projectName,
      'label': selectedValue,
      'n': imageNum
    }
    console.log(sentData)
    try {
      const response = await fetch('http://127.0.0.1:5000/api/req_dataset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sentData),
      });
      const result = await response.json();
      setImages(result.images);
    } catch (error) {
      console.error('no image', error)
    }
    
  }
  return (
    <div className='preview-wrapper'>
      <div className='preview-config-wrapper'>
        <div className='preview-config-field'>
          <p>数量</p>
          <input
            type='number'
            min="1"
            max="100"
            value={imageNum}
            onChange={handleChange}
          />
          <p>Label</p>
          <select value={selectedValue} onChange={handleSelectedChange}>
            {labelList.map((label, index) => (
              <option key={index} value={label}>
                {label}
              </option>
            ))}
          </select>
          <div className='search-button' onClick={handleGetImage} style={{ cursor: 'pointer' }}>
            <SearchIcon className='search-icon' />
          </div>
        </div>
      </div>
      <div className='preview-image-field'>
        {images && images.map((value, index) => (
          <div key={index}>
            <PreviewImage image={value} />
          </div>
        ))}
      </div>
    </div>
  )
}

export default ProjectPreview
