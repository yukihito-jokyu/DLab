import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const CSVUploader = ({ onCSVUpload }) => {
  const onDrop = useCallback((acceptedFiles) => {
    // ドロップされたファイルを処理
    acceptedFiles.forEach((file) => {
      // 拡張子が .csv 以外の場合はエラーメッセージをログに出力
      if (!file.name.endsWith('.csv')) {
        console.error(`Invalid file format: ${file.name}. Only CSV files are allowed.`);
        return;
      }

      const reader = new FileReader();

      reader.onload = () => {
        const csvData = reader.result;
        // 読み込んだCSVデータを親コンポーネントに渡す
        onCSVUpload(csvData);
      };

      reader.readAsText(file);
    });
  }, [onCSVUpload]);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div {...getRootProps()} style={dropzoneStyle}>
      <input {...getInputProps()} />
      <p>CSVファイルをドラッグ＆ドロップまたはクリックしてアップロード</p>
    </div>
  );
};

const dropzoneStyle = {
  border: '2px dashed #cccccc',
  borderRadius: '4px',
  padding: '20px',
  textAlign: 'center',
  cursor: 'pointer',
};

export default CSVUploader;
