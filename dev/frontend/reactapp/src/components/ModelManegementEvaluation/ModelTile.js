import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import './ModelManegementEvaluation.css';
import { ReactComponent as PictureIcon } from '../../assets/svg/graph_24.svg';
import { useNavigate, useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';
import useFetchStatus from '../../hooks/useFetchStatus';
import useFetchAccuracyAndLoss from '../../hooks/useFetchAccuracyAndLoss';
import DisplayResult from './DisplayResult';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function ModelTile({ modelName, date, isChecked, modelId, checkBoxChange, status, userId }) {
  const { projectName } = useParams();
  const [isPicture, setIsPicture] = useState(false);
  const [isExpanding, setIsExpanding] = useState(false);
  const [tileColer, setTileColer] = useState();
  const [tileHeight, setTileHeight] = useState('70px');
  const [isHover, setIsHover] = useState();
  const navigate = useNavigate();

  const { accuracyData, lossData } = useFetchTrainingResults(modelId);
  const { accuracy, loss } = useFetchAccuracyAndLoss(modelId);
  const currentStatus = useFetchStatus(modelId);

  useEffect(() => {
    const initTileColer = () => {
      if (currentStatus === 'pre') {
        const color = {
          backgroundColor: 'rgb(80, 166, 255, 0.15)',
        };
        setTileColer(color);
      } else if (currentStatus === 'doing') {
        const color = {
          background: 'linear-gradient(91.27deg, rgba(196, 73, 255, 0.5) 0.37%, rgba(71, 161, 255, 0.5) 99.56%)',
        };
        setTileColer(color);
      } else if (currentStatus === 'done') {
        const color = {
          backgroundColor: 'rgba(39, 203, 124, 0.5)',
        };
        setTileColer(color);
      }
    };
    initTileColer();
  }, [currentStatus]);

  useEffect(() => {
    const handleTransitionEnd = () => {
      if (!isExpanding) {
        setIsPicture(false);
      }
    };

    const tileElement = document.querySelector('.model-tile-wrapper');
    if (tileElement) {
      tileElement.addEventListener('transitionend', handleTransitionEnd);
    }

    return () => {
      if (tileElement) {
        tileElement.removeEventListener('transitionend', handleTransitionEnd);
      }
    };
  }, [isExpanding]);

  useEffect(() => {
    setTileHeight(isExpanding ? '500px' : '70px');
  }, [isExpanding]);

  const formatTimestamp = (timestamp) => {
    if (timestamp && timestamp.seconds) {
      const date = new Date(timestamp.seconds * 1000);
      return date.toLocaleDateString(); // 日付と時刻をローカル形式で表示
    }
    return '';
  };

  const handleClick = () => {
    if (currentStatus !== 'pre') {
      if (isPicture) {
        setIsExpanding(false);
      } else {
        setIsPicture(true);
        setIsExpanding(true);
      }
    }
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
    <div className='model-tile-wrapper' style={{ ...tileColer, height: tileHeight }} onMouseEnter={() => setIsHover(true)} onMouseLeave={() => setIsHover(false)}>
      {isHover && (
        <div>
          {currentStatus === 'pre' && isPicture === false && (
            <div className='cursor-tooltip1'>
              <p>学習前</p>
            </div>
          )}
          {currentStatus === 'doing' && isPicture === false && (
            <div className='cursor-tooltip2'>
              <p>学習中</p>
            </div>
          )}
          {currentStatus === 'done' && isPicture === false && (
            <div className='cursor-tooltip3'>
              <p>学習済み</p>
            </div>
          )}
        </div>
      )}
      <div className='model-title-field'>
        <div className='model-check-box-wrapper'>
          <label className="custom-checkbox">
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => checkBoxChange(modelId)}
            />
            <span className="checkmark"></span>
          </label>
        </div>
        <div className='model-title' onClick={handleNav} style={{ cursor: 'pointer' }}>
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
          <div
            className='model-picture-icon-wrapper'
            onClick={handleClick}
            style={{
              cursor: currentStatus === 'pre' ? 'not-allowed' : 'pointer',
              opacity: currentStatus === 'pre' ? 0.6 : 1
            }}
          >
            <PictureIcon className='picture-svg' />
          </div>
        </div>
      </div>
      {isPicture && accuracyData && lossData && accuracyData.labels && lossData.labels &&
        <div className='graph-field'>
          <div className='model-picture-filed-wrapper'>
            <DisplayResult data={accuracyData} type="Accuracy" showTitle={true} />
            <DisplayResult data={lossData} type="Loss" showTitle={true} />
          </div>
        </div>
      }
    </div>
  );
}

export default ModelTile;
