import React, { useCallback, useState } from 'react';
import './ImageUploadModal.css';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';
import { useDropzone } from 'react-dropzone';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import { uploadUserImage } from '../../db/function/storage';

function ImageUploadModal({ deleteModal }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const style = {
    fontSize: "30px",
    fontWeight: "600"
  }
  const text = '画像をアップロード';
  const text2 = '保存';

  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageType, setImageType] = useState('');
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['jpg', 'jpeg', 'png', 'gif'];
  
    if (!allowedExtensions.includes(fileExtension)) {
      setError('許可されていないファイル形式です。');
      return;
    }
    setImageType(fileExtension);
    setImageFile(file)
    const reader = new FileReader();

    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (img.width === 400 && img.height === 400) {
          console.log(`画像サイズ: ${img.width} x ${img.height} ピクセル`);
          setImage(e.target.result);
          setError(null);
        } else {
          console.log(`不適切な画像サイズ: ${img.width} x ${img.height} ピクセル`);
          setImage(null);
          setImageFile(null)
          setError('画像サイズは400x400ピクセルである必要があります。');
        }
      };
      img.src = e.target.result;
    };

    reader.readAsDataURL(file);
  }, []);

  const onDropRejected = useCallback((rejectedFiles) => {
    setError('不適切なファイル形式です。画像ファイルをアップロードしてください。');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'image/png': ['.png'],
      'image/jpg': ['.jpg'],
      'image/jpeg': ['.jpeg'],
    },
    multiple: false
  });
  const handleUpload = async () => {
    deleteModal()
    await uploadUserImage(userId, imageFile, imageType)
  }
  return (
    <div>
      <div className='image-upload-modal-wrapper'></div>
      <div className='image-upload-modal-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='image-upload-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={text} style={style} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='image-upload-comment'>
                <div {...getRootProps()} className='drag-and-drop-field'>
                  <input {...getInputProps()} />
                  <div className='drag-and-drop-comment-field'>
                    {!image &&
                    <div>
                      {isDragActive ? (
                        <p>ここにドロップしてください...</p>
                      ) : (
                        <p>ここをクリックするか、<br/>画像をドラッグ&ドロップしてください<br/>画像サイズは400×400のみ対応</p>
                      )}
                      </div>
                    }
                    {image &&
                    <div className='uploaded-image-wrapper'>
                      <img src={image} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '300px' }} />
                      <div className='delet-icon-wrapper'>
                        <div onClick={() => setImage(null)} className='image-delet-svg-wrapper'>
                          <DeletIcon className='image-delet-svg' />
                        </div>
                      </div>
                    </div>
                    }
                  </div>
                  {error && <p style={{ color: 'red' }}>{error}</p>}
                  
                </div>
              </div>
              <div className='alert-modal'>
                {image && <div onClick={handleUpload}>
                  <GradationButton text={text2} />
                </div>}
              </div>
              <div className='train-modal-delet-button-field' onClick={deleteModal}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageUploadModal
