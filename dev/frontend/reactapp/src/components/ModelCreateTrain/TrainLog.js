import React, { useEffect, useState } from 'react';
import './TrainLog.css';
import TrainPanelTital from './TrainPanelTital';
import { socket } from '../../socket/socket';
import { useParams } from 'react-router-dom';

function TrainLog() {
  const [log, setLog] = useState('');
  const { modelId } = useParams();
  // socket通信にて
  useEffect(() => {
    const handleTrainResults = (response) => {
      console.log('Response from server:', response.Epoch, response.TrainAcc, response.ValAcc, response.TrainLoss, response.ValLoss);
      const getLog = 'epoch: ' + response.Epoch + ' Train Accuracy: ' + response.TrainAcc + ' Valid Accuracy: ' + response.ValAcc + ' Train Loss: ' + response.TrainLoss + ' Valid Loss: ' + response.ValLoss;
      setLog((prevText) => prevText + '\n' + getLog)
    }

    socket.on('train_image_results'+modelId, handleTrainResults);

    // クリーンアップ
    return () => {
      socket.off('image_train_end'+modelId, handleTrainResults);
    }
  }, [modelId])
  return (
    <div className='train-log-wrapper'>
      <TrainPanelTital />
      <div className='log-field'>
        <p dangerouslySetInnerHTML={{ __html: log }} />
      </div>
    </div>
  )
}

export default TrainLog
