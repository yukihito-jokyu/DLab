import React, { useEffect, useState } from 'react';
import './TrainPanel.css';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';
import InformationModal from './Modal/InformationModal';

function TrainPanel({ trainInfo, setTrainInfo }) {
  const [sortTrainInfo, setSortTrainInfo] = useState(null);
  const [information, setInformation] = useState(false);
  const [paramName, setParamName] = useState('');


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


  const handleChangeParameter = (key, value) => {
    const newParameter = { ...trainInfo };
    newParameter[key] = value;
    setTrainInfo(newParameter);
  };

  return (
    <>
      <div className="train-panel-wrapper">
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
    </>
  );
}

export default TrainPanel;