import React, { useState } from 'react'

function CheckImage() {
  const [trainImage, setTrainImage] = useState([]);
  const [testImage, setTestImage] = useState([]);

  const handleGetImage = async () => {
    const response = await fetch('http://127.0.0.1:5000/Image/LoadImage', {
      method: 'POST',
    });
    const result = await response.json();
    setTrainImage(result.train_image_list);
    setTestImage(result.test_image_list);
  }
  return (
    <div>
      <h1>CheckImage</h1>
      <button onClick={handleGetImage}>Image取得</button>
      <div>
        <p>train</p>
        <div
          style={{
            display: 'flex'
          }}
        >
          {trainImage.map((image, index) => (
            <img key={index} src={`data:image/png;base64,${image}`} alt="train_Image" />
          ))}
        </div>
      </div>
      <div>
        <p>test</p>
        <div
          style={{
            display: 'flex'
          }}
        >
          {testImage.map((image, index) => (
            <img key={index} src={`data:image/png;base64,${image}`} alt="train_Image" />
          ))}
        </div>
      </div>
    </div>
  )
}

export default CheckImage
