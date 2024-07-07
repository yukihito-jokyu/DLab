import React, { useEffect, useState } from 'react';
import './TrainPanel.css';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';

function TrainPanel({ trainInfo, setTrainInfo }) {
  const [sortTrainInfo, setSortTrainInof] = useState(null);
  useEffect(() => {
    const sortObject = () => {
      const sortedKeys = Object.keys(trainInfo).sort();
      
      // ソートされたキーを使って新しいオブジェクトを作成
      const sortedObj = {};
      for (const key of sortedKeys) {
        sortedObj[key] = trainInfo[key];
      }
      setSortTrainInof(sortedObj);
    };
    if (trainInfo) {
      sortObject();
    }
  }, [trainInfo]);
  const handleChangeParameter = (key, value) => {
    const newParameter = { ...trainInfo };
    newParameter[key] = value;
    setTrainInfo(newParameter)
  }
  return (
    <div className='train-panel-wrapper'>
      <TrainPanelTital />
      <div className='panel-field'>
        {sortTrainInfo && Object.entries(sortTrainInfo).map(([key, value], index) => (
          <div key={index}>
            <TrainPanelEdit parameter={key} value={value} handleChangeParameter={handleChangeParameter} />
          </div>
        ))}
      </div>
    </div>
  )
}

export default TrainPanel
