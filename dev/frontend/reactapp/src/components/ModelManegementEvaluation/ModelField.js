import React, { useEffect, useState } from 'react';
import './ModelManegementEvaluation.css';
import ModelFieldHeader from './ModelFieldHeader';
import ModelTile from './ModelTile';
import ModelCreateButton from './ModelCreateButton';
import DLButton from './DLButton';
import { useNavigate, useParams } from 'react-router-dom';
import { deleteModels } from '../../db/function/model_management';
import ModelCreateField from './ModelCreateField';
import AlertModal from '../utils/AlertModal';
import { sendEmailVerification } from 'firebase/auth';
import { getModelId } from '../../db/function/model_management';
import { deleteFilesInDirectory } from '../../db/function/storage';

function ModelField() {
  const [models, setModels] = useState([]);
  const [DL, setDL] = useState();
  const [DLModal, setDLModal] = useState(false);
  const [modelDeleteModal, setModelDeleteModal] = useState(false);
  const [create, setCreate] = useState(false);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const { projectName } = useParams();
  // console.log(projectName)
  useEffect(() => {
    const fetchProjects = async () => {
      const dataList = await getModelId(userId, projectName);
      if (dataList !== null) {
        const modelsWithCheckbox = dataList.map(model => ({ ...model, isChecked: false }));
        setModels(modelsWithCheckbox);
      };
    };
    fetchProjects();
  }, [create, userId, projectName]);

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
      Project_name: projectName,
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

  const handleDeleteStorage = async () => {
    const checkedModels = models.filter(model => model.isChecked);
    const deletePromises = checkedModels.map(async (model) => {
      const deletePath = `user/${userId}/${projectName}/${model.model_id}`
      await deleteFilesInDirectory(deletePath);
    })
    try {
      // すべての削除処理が完了するのを待つ
      await Promise.all(deletePromises);
      console.log("All selected files have been deleted successfully.");
    } catch (error) {
      console.error("Error deleting files:", error);
    }
  }

  const handleDelateModel = () => {
    // firebaseのdatabase内のデータ削除とバックエンドの削除
    handleDelate();
    // fireStorage内のデータ削除
    handleDeleteStorage();
    setModelDeleteModal(!modelDeleteModal);
  }
  const sendText2 = 'チェックしたモデルを削除しますか？'

  // zipファイルのダウンロード
  const handleDownload = async () => {
    try {
      const checkedModels = models.filter(model => model.isChecked);
      const modelIdList = checkedModels
        .map(item => item.model_id);
      const sentData = {
        user_id: userId,
        Project_name: projectName,
        model_id_list: modelIdList
      }
      const response = await fetch('http://127.0.0.1:5000/download_zip', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sentData),
      });
      console.log(response)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      

      const blob = await response.blob();

      const link = document.createElement('a');
      link.href = window.URL.createObjectURL(blob);
      link.download = `models_${new Date().toISOString()}.zip`;
      link.click();

      window.URL.revokeObjectURL(link.href);
    } catch (e) {
      console.error('Download failed', e)
    }
    
  }

  const handleDownloadZip = async () => {
    const checkedModels = models.filter(model => model.isChecked);
    const downloadModels = checkedModels.filter(model => (model.status === 'done'));
    console.log(downloadModels);
    

  }

  // DLモーダル表示非表示
  const changeDLModal = () => {
    setDLModal(!DLModal);
  }
  // DL関数
  const getDLItem = async () => {
    setDLModal(!DLModal);
    await handleDownloadZip();
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
                status={model.status}
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
