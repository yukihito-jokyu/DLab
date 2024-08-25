import React, { useEffect, useState } from 'react';
import './ModelManegementEvaluation.css';
import ModelFieldHeader from './ModelFieldHeader';
import ModelTile from './ModelTile';
import ModelCreateButton from './ModelCreateButton';
import DLButton from './DLButton';
import { useParams } from 'react-router-dom';
import { deleteModels } from '../../db/function/model_management';
import ModelCreateField from './ModelCreateField';
import AlertModal from '../utils/AlertModal';
import { getModelId } from '../../db/function/model_management';
import { saveAs } from 'file-saver';
import { combineZipsIntoOne, createZipFromDirectory, deleteFilesInDirectory } from '../../db/function/storage';

function ModelField() {
  const [models, setModels] = useState([]);
  const [DL, setDL] = useState();
  const [DLModal, setDLModal] = useState(false);
  const [modelDeleteModal, setModelDeleteModal] = useState(false);
  const [create, setCreate] = useState(false);
  const [successDeleteModal, setSuccessDeleteModal] = useState(false);
  const [successModelCreate, setSuccessModelCreate] = useState(false);
  const [successModelDownload, setSuccessModelDownload] = useState(false);
  const [sameModelName, setSameModelName] = useState(false);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const { projectName } = useParams();
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
      model.id === id ? { ...model, isChecked: !model.isChecked } : model
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
  };

  // モデル削除モーダルの表示非表示
  const changeModelDeleteModal = () => {
    const isCheckedExists = models.some(model => model.isChecked);
    if (isCheckedExists) {
      setModelDeleteModal(!modelDeleteModal);
    }
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
      setSuccessDeleteModal(true);
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

  // 日時を含むファイル名を生成する関数
  const generateFileNameWithDateTime = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');

    return `combined_models_${year}${month}${day}_${hours}${minutes}${seconds}.zip`;
  };

  // ZIPファイルをダウンロードする関数
  const handleDownloadZip = async () => {
    const checkedModels = models.filter(model => model.isChecked);
    const downloadModels = checkedModels.filter(model => model.status === 'done');
    try {
      const zipPromises = downloadModels.map(model => createZipFromDirectory(`user/${userId}/${projectName}/${model.model_id}`));

      // すべてのZIPファイルを生成
      const zipBlobs = await Promise.all(zipPromises);

      // すべてのZIPファイルを1つにまとめる
      const combinedZipBlob = await combineZipsIntoOne(zipBlobs, downloadModels);

      // 現在の日時を取得し、ファイル名を生成
      const now = new Date();
      const fileName = generateFileNameWithDateTime(now);

      // まとめたZIPファイルをダウンロード
      saveAs(combinedZipBlob, fileName);

      console.log("All selected files have been combined and downloaded successfully.");
      setSuccessModelDownload(true);
    } catch (error) {
      console.error("Error during ZIP creation and download:", error);
    }
  };

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
                date={model.date}
                isChecked={model.isChecked}
                modelId={model.id}
                checkBoxChange={handleCheckboxChange}
                status={model.status}
                userId={userId}
              />
            </div>
          ))
        ) : (<></>)}
        <ModelCreateButton handleCreateModal={handleCreateModal} />
      </div>
      {DL ? (
        <div className='DL-field' onClick={changeDLModal} style={{ cursor: 'pointer' }}>
          <DLButton />
        </div>
      ) : (
        <></>
      )
      }
      {create ? (
        <div className='create-background-field'>
          <ModelCreateField handleCreateModal={handleCreateModal} setSuccessModelCreate={setSuccessModelCreate} setSameModelName={setSameModelName} />
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
      {successDeleteModal && <AlertModal deleteModal={() => setSuccessDeleteModal(false)} handleClick={() => setSuccessDeleteModal(false)} sendText={'チェックしたモデルが削除されました'} />}
      {successModelCreate && <AlertModal deleteModal={() => setSuccessModelCreate(false)} handleClick={() => setSuccessModelCreate(false)} sendText={'モデルが作成されました'} />}
      {successModelDownload && <AlertModal deleteModal={() => setSuccessModelDownload(false)} handleClick={() => setSuccessModelDownload(false)} sendText={'モデルがダウンロードされました'} />}
      {sameModelName && <AlertModal deleteModal={() => setSameModelName(false)} handleClick={() => setSameModelName(false)} sendText={'モデル名が重複しています'} />}
    </div>
  )
}

export default ModelField;
