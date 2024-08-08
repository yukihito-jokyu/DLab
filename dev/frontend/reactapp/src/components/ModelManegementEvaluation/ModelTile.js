import React, { useCallback, useEffect, useState } from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as PictureIcon } from '../../assets/svg/graph_24.svg'
import { useNavigate, useParams } from 'react-router-dom';
import { getImage } from '../../db/function/storage';

function ModelTile({ modelName, accuracy, loss, date, isChecked, modelId, checkBoxChange, status  }) {
  const { projectName } = useParams()
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [isPicture, setIsPicture] = useState(false);
  const [tileColer, setTileColer] = useState();
  const [isHover, setIsHover] = useState();
  const [accuracyImage, setAccuracyImage] = useState();
  const [lossImage, setLossImage] = useState();
  const navigate = useNavigate();

  // 学習曲線の取得
  useEffect(() => {
    const fetchImage = async () => {
      const accuracyPath = `user/${userId}/${projectName}/${modelId}/accuracy_curve.png`
      const accuracyImageURL = await getImage(accuracyPath);
      const lossPath = `user/${userId}/${projectName}/${modelId}/loss_curve.png`
      const lossImageURL = await getImage(lossPath);
      setAccuracyImage(accuracyImageURL);
      setLossImage(lossImageURL)
    }
    if (status === 'done') {
      fetchImage()
    }
  }, [userId, projectName, modelId, status]);

  useEffect(() => {
    const initTileColer = () => {
      if (status === 'pre') {
        const color = {
          backgroundColor: 'rgb(80, 166, 255, 0.15)'
        }
        setTileColer(color);
      } else if (status === 'doing') {
        const color = {
          background: 'linear-gradient(91.27deg, rgba(196, 73, 255, 0.5) 0.37%, rgba(71, 161, 255, 0.5) 99.56%)'
        }
        setTileColer(color);
      } else if (status === 'done') {
        const color = {
          backgroundColor: 'rgba(39, 203, 124, 0.5)'
        }
        setTileColer(color);
      }
    }
    initTileColer();
  }, [status]);

  const formatTimestamp = (timestamp) => {
    if (timestamp && timestamp.seconds) {
      const date = new Date(timestamp.seconds * 1000);
      return date.toLocaleDateString(); // 日付と時刻をローカル形式で表示
    }
    return '';
  };
  const handleClick = () => {
    setIsPicture(!isPicture);
  };
  const handleNav = () => {
    sessionStorage.setItem('modelId', JSON.stringify(modelId));
    navigate(`/ModelCreateTrain/${projectName}/${modelId}`);
  };

  const TextDisplay = ({ text, maxLength }) => {
    const displayedText = text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
    
    return <div className="text-container"><p>{displayedText}</p></div>;
  };



  return (
    <div className='model-tile-wrapper'
      style={tileColer}
      onMouseEnter={() => setIsHover(true)}
      onMouseLeave={() => setIsHover(false)}
    >
      {isHover && (
        <div>
          {status === 'pre' && (
            <div className='cursor-tooltip1'>
              <p>学習前</p>
            </div>
          )}
          {status === 'doing' && (
            <div className='cursor-tooltip2'>
              <p>学習中</p>
            </div>
          )}
          {status === 'done' && (
            <div className='cursor-tooltip3'>
              <p>学習済み</p>
            </div>
          )}
        </div>
      )}
      <div className='model-title-field'>
        <div className='model-check-box-wrapper'>
          {/* <div className='model-check-box'></div> */}
          <label className="custom-checkbox">
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => checkBoxChange(modelId)}
            />
            <span className="checkmark"></span>
          </label>
        </div>
        <div className='model-title' onClick={handleNav}>
          <TextDisplay text={modelName} maxLength={20} />
        </div>
        <div className='model-accuracy'>
          <p>{accuracy}</p>
        </div>
        <div className='model-loss'>
          <p>{loss}</p>
        </div>
        <div className='model-date'>
          <p>{formatTimestamp(date)}</p>
        </div>
        <div className='model-picture'>
          <div className='model-picture-icon-wrapper' onClick={handleClick}>
            <PictureIcon className='picture-svg' />
          </div>
        </div>
      </div>
      {isPicture && 
        <div className='graph-field'>
          <p>Graph</p>
          <div className='model-picture-filed-wrapper'>
            <div className='model-accuracy-picture'>
              {accuracyImage ? (
                <img src={accuracyImage} alt='firebase Storage error' />
              ) : (
                <p>No Image</p>
              )}
            </div>
            <div className='model-loss-picture'>
              {lossImage ? (
                <img src={lossImage} alt='firebase Storage error' />
              ) : (
                <p>No Image</p>
              )}
            </div>
          </div>
        </div>
      }
      
    </div>
  )
}

export default ModelTile
