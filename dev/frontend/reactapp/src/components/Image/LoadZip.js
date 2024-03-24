import React, { useState } from 'react'

function LoadZip() {
  const [trainZip, setTrainZip] = useState(null);
  const [testZip, setTestZip] = useState(null);

  const handleTrainZipChange = (e) => {
    setTrainZip(e.target.files[0]);
  };
  const handleTestZipChange = (e) => {
    setTestZip(e.target.files[0]);
  };

  const uploadTrainZip = async () => {
    const formData = new FormData();
    formData.append('file', trainZip);
    const response = await fetch('http://127.0.0.1:5000/Image/UploadZip/train', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    console.log(result);
  };
  const uploadTestZip = async () => {
    const formData = new FormData();
    formData.append('file', testZip);
    const response = await fetch('http://127.0.0.1:5000/Image/UploadZip/test', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    console.log(result);
  };

  // zip解凍
  const handleUnzZip = async () => {
    const response = await fetch('http://127.0.0.1:5000/Image/UnZip', {
      method: 'POST',
    });
    const result = await response.json();
    console.log(result);
  }
  return (
    <div>
      <div>
        <input type='file' onChange={handleTrainZipChange} />
        <button onClick={uploadTrainZip}>train zip Upload</button>
      </div>
      <div>
        <input type='file' onChange={handleTestZipChange} />
        <button onClick={uploadTestZip}>test zip Upload</button>
      </div>
      <div>
        <p>zipファイル解凍</p>
        <button onClick={handleUnzZip}>UnZip</button>
      </div>
    </div>
  )
}

export default LoadZip
