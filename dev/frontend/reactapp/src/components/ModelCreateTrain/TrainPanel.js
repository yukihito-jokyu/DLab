import React, { useEffect, useState } from 'react';
import './TrainPanel.css';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';
import InformationModal from './Modal/InformationModal';
import { useParams } from 'react-router-dom';

function TrainPanel({ trainInfo, setTrainInfo, augmentationParams, setAugmentationParams }) {
  const { task } = useParams();
  const [sortTrainInfo, setSortTrainInfo] = useState(null);
  const [sortAugmentationParams, setSortAugmentationParams] = useState(null);
  const [information, setInformation] = useState(false);
  const [paramName, setParamName] = useState('');

  // 高さを動的に設定
  const wrapperClassName = task === 'ImageClassification' ? 'train-panel-wrapper' : 'train-panel-wrapper full-height';

  // 学習パラメータのソート
  useEffect(() => {
    const sortObject = () => {
      const sortedKeys = Object.keys(trainInfo).sort();
      const sortedObj = {};
      for (const key of sortedKeys) {
        sortedObj[key] = trainInfo[key];
      }
      setSortTrainInfo(sortedObj);
    };
    if (trainInfo) {
      sortObject();
    }
  }, [trainInfo]);

  // データ拡張パラメータのソート
  useEffect(() => {
    const sortObject = () => {
      const sortedKeys = Object.keys(augmentationParams).sort();
      const sortedObj = {};
      for (const key of sortedKeys) {
        sortedObj[key] = augmentationParams[key];
      }
      setSortAugmentationParams(sortedObj);
    };
    if (augmentationParams) {
      sortObject();
    }
  }, [augmentationParams]);

  const handleChangeParameter = (key, value) => {
    const newParameter = { ...trainInfo };
    newParameter[key] = value;
    setTrainInfo(newParameter);
  };

  const handleChangeAugmentationParameter = (key, value) => {
    const newParams = { ...augmentationParams };
    newParams[key] = value;
    setAugmentationParams(newParams);
  };

  return (
    <>
      <div className={wrapperClassName}>
        <TrainPanelTital title={'学習パラメータ'} />
        <div className='panel-field'>
          {sortTrainInfo && Object.entries(sortTrainInfo).map(([key, value], index) => (
            <div key={index}>
              <TrainPanelEdit
                parameter={key}
                value={value}
                handleChangeParameter={handleChangeParameter}
                setInformation={setInformation}
                setParamName={setParamName}
              />
            </div>
          ))}
        </div>
        {information && <InformationModal infoName={paramName} handleDelete={setInformation} />}
      </div>

      {task === 'ImageClassification' && (
        <div className='train-panel-wrapper'>
          <TrainPanelTital title={'データ拡張パラメータ'} />
          <div className='panel-field'>
            {sortAugmentationParams && Object.entries(sortAugmentationParams).map(([key, value], index) => (
              <div key={index}>
                <TrainPanelEdit
                  parameter={key}
                  value={value}
                  handleChangeParameter={handleChangeAugmentationParameter}
                  setInformation={setInformation}
                  setParamName={setParamName}
                />
              </div>
            ))}
          </div>
          {information && <InformationModal infoName={paramName} handleDelete={setInformation} />}
        </div>
      )}
    </>
  );
}

export default TrainPanel;