import React, { useState } from 'react';
import './Community.css';
import PreviewImage from './PreviewImage';
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg';
import { useParams } from 'react-router-dom';
import { PropagateLoader } from 'react-spinners';

function ProjectPreview({ labelList }) {
  const { projectName } = useParams();
  const [selectedValue, setSelectedValue] = useState(labelList[0]);
  const [imageNum, setImageNum] = useState(1);
  const [images, setImages] = useState();
  const [loading, setLoading] = useState(false);

  const handleSelectedChange = (e) => {
    setSelectedValue(e.target.value);
  };

  const handleChange = (event) => {
    const inputValue = event.target.value;
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

  const handleGetImage = async () => {
    setLoading(true);
    const sentData = {
      'dataset': projectName,
      'label': selectedValue,
      'n': imageNum
    };
    console.log(sentData);
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
      console.error('no image', error);
    } finally {
      setLoading(false);
    }
  };

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
        {loading ? (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
          }}>
            <PropagateLoader color="linear-gradient(91.27deg, #C49EFF 0.37%, #47A1FF 99.56%)" size={20} />
          </div>
        ) : (
          images && images.map((value, index) => (
            <div key={index}>
              <PreviewImage image={value} />
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ProjectPreview;
