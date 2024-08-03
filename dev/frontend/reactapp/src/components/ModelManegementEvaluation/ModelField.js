import React, { useEffect, useState } from 'react';
import './ModelManegementEvaluation.css';
import ModelFieldHeader from './ModelFieldHeader';
import ModelTile from './ModelTile';
import ModelCreateButton from './ModelCreateButton';
import DLButton from './DLButton';
import { useNavigate } from 'react-router-dom';
import { deleteModels, getModelId } from '../../db/firebaseFunction';
import ModelCreateField from './ModelCreateField';
import AlertModal from '../utils/AlertModal';
import { sendEmailVerification } from 'firebase/auth';

function ModelField() {
  const [models, setModels] = useState([]);
  const [DL, setDL] = useState();
  const [DLModal, setDLModal] = useState(false);
  const [modelDeleteModal, setModelDeleteModal] = useState(false);
  const [create, setCreate] = useState(false);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const projectId = JSON.parse(sessionStorage.getItem('projectId'));
  useEffect(() => {
    const fetchProjects = async () => {
      const dataList = await getModelId(userId, projectId);
      if (dataList !== null) {
        const modelsWithCheckbox = dataList.map(model => ({ ...model, isChecked: false }));
        setModels(modelsWithCheckbox);
      };
    };
    fetchProjects();
  }, [create, userId, projectId]);

  // 照準降順並び替え
  const accuracySort = (isAscending) => {
    const sortModels = [...models].sort((a, b) => {
      return isAscending ? b.accuracy - a.accuracy : a.accuracy - b.accuracy;
    });
    setModels(sortModels);
  };
  const lossSort = (isAscending) => {
    const sortModels = [...models].sort((a, b) => {
      return isAscending ? b.loss - a.loss : a.loss - b.loss;
    });
    setModels(sortModels);
  };
  const dateSort = (isAscending) => {
    const sortModels = [...models].sort((a, b) => {
      return isAscending ? b.date - a.date : a.date - b.date;
    });
    setModels(sortModels);
  };

  // チェックボックスの更新
  const handleCheckboxChange = (id) => {
    const updateModels = models.map(model =>
      model.id === id ? {...model, isChecked: !model.isChecked} : model
    );
    setModels(updateModels);
  };

  // modelsが更新された後に実行されるコード
  useEffect(() => {
    const judgeDL = models.some(item => item.status === 'done' && item.isChecked);
    setDL(judgeDL);
  }, [models]);

  // モデル作成モーダル表示非表示
  const handleCreateModal = () => {
    setCreate(!create);
  };

  // モデル削除
  const handleDelate = async () => {
    const checkedModels = models.filter(model => model.isChecked);
    const deletePromises = checkedModels.map(model => deleteModels(model.model_id));
    await Promise.all(deletePromises);
    const remainingModels = models.filter(model => !model.isChecked);
    setModels(remainingModels);
    const modelIdList = checkedModels
      .map(item => item.model_id);
    const sentData = {
      user_id: userId,
      Project_name: projectId,
      model_id_list: modelIdList
    }
    const response = await fetch('http://127.0.0.1:5000/del_dir/model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sentData),
    });
    const result = await response.json();
    console.log(result);
  };
  // モデル削除モーダルの表示非表示
  const changeModelDeleteModal = () => {
    setModelDeleteModal(!modelDeleteModal);
  };
  const handleDelateModel = () => {
    handleDelate();
    setModelDeleteModal(!modelDeleteModal);
  }
  const sendText2 = 'チェックしたモデルを削除しますか？'

  // DLモーダル表示非表示
  const changeDLModal = () => {
    setDLModal(!DLModal)
  }
  // DL関数
  const getDLItem = () => {
    setDLModal(!DLModal)
  }
  const sendText = 'チェックしたモデルをダウンロードしますか？'
  return (
    <div className='model-field-wrapper'>
      <ModelFieldHeader
        accuracySort={accuracySort}
        lossSort={lossSort}
        dateSort={dateSort}
        handleDelate={changeModelDeleteModal}
      />
      <div className='tile-field'>
        {models.length > 0 ? (
          models.map((model) => (
            <div key={model.id}>
              <ModelTile 
                modelName={model.model_name}
                accuracy={model.accuracy}
                loss={model.loss}
                date={model.date}
                isChecked={model.isChecked}
                modelId={model.id}
                checkBoxChange={handleCheckboxChange}
              />
            </div>
          ))
        ) : (<></>)}
        <ModelCreateButton handleCreateModal={handleCreateModal} />
      </div>
      {DL ? (
        <div className='DL-field' onClick={changeDLModal}>
          <DLButton />
        </div>
      ) : (
        <></>
      )
      }
      {create ? (
        <div className='create-background-field'>
          <ModelCreateField handleCreateModal={handleCreateModal} />
        </div>
      ) : (
        <></>
      )}
      {DLModal ? (
        <div>
          <AlertModal deleteModal={changeDLModal} handleClick={getDLItem} sendText={sendText} />
        </div>
      ) : (
        <></>
      )}
      {modelDeleteModal ? (
        <div>
          <AlertModal deleteModal={changeModelDeleteModal} handleClick={handleDelateModel} sendText={sendText2} />
        </div>
      ) : (
        <></>
      )}
    </div>
  )
}

export default ModelField;
