import React, { useEffect, useState } from 'react'
import './ModelCreateTrain.css'
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';
import { getModelStructure, getTrainInfo } from '../../db/firebaseFunction';
import { socket } from '../../socket/socket';

function TrainModal({ changeTrain, flattenShape }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const projectId = JSON.parse(sessionStorage.getItem('projectId'));
  const modelId = JSON.parse(sessionStorage.getItem('modelId'));
  
  const [train, setTrain] = useState(false);
  const text = '学習用ファイルを作成してください'
  const text2 = '作成する'
  const text3 = '学習が可能になりました'
  const text4 = '学習する'
  const style = {
    fontSize: "26px",
    fontWeight: "600"
  }
  const handleMakeConfig = async () => {
    setTrain(true);
    const structure = await getModelStructure(modelId);
    const sentData = {
      user_id: userId,
      project_name: projectId,
      model_id: modelId,
      structure: structure,
      flattenshape: flattenShape
    };
    const response = await fetch('http://127.0.0.1:5000/ImageClassification/make/config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sentData),
    });
    console.log(response)
  };

  // 学習開始(socket)
  const startTrain = async () => {
    if (socket) {
      const trainInfo = await getTrainInfo(modelId)
      const sentData = {
        user_id: userId,
        project_name: projectId,
        model_id: modelId,
        Train_info: trainInfo
      }
      socket.emit('image_train_start', sentData);
    }
  }

  // socket通信にて
  useEffect(() => {
    const handleTrainResults = (response) => {
      console.log('Response from server:', response.Epoch, response.TrainAcc, response.ValAcc, response.TrainLoss, response.ValLoss)
    }

    socket.on('train_image_results'+modelId, handleTrainResults);

    // クリーンアップ
    return () => {
      socket.off('image_train_end'+modelId, handleTrainResults);
    }
  }, [modelId])
  return (
    <div>
      <div className='train-modal-wrapper'></div>
      {!train && (<div className='tile-add-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='train-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={text} style={style} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='train-modal'>
                <div onClick={handleMakeConfig}>
                  <GradationButton text={text2} />
                </div>
              </div>
              <div className='train-modal-delet-button-field' onClick={changeTrain}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>)}
      {train && (<div className='tile-add-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='train-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={text3} style={style} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='train-modal' onClick={startTrain}>
                <GradationButton text={text4} />
              </div>
              <div className='train-modal-delet-button-field' onClick={changeTrain}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>)}
    </div>
  )
}

export default TrainModal