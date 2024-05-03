import React from 'react';
import '../css/CreateDataset.css';

function CreateDataset() {
  return (
    <div className='create-dataset-wrapper'>
      <div className='create-dataset-name'>
        <p>Datasets</p>
      </div>
      <div className='create-dataset-upload'>
        <p>クリック・ドラッグアウトでアップロード</p>
      </div>
      <div className='create-dataset-upload-name'>
        <p>✔ [FileName.zip]</p>
      </div>
    </div>
  )
}

export default CreateDataset
