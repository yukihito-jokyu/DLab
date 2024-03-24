import React from 'react';

function CartPoleDownload() {
  const handlepthDownload = async () => {
    const response = await fetch('http://127.0.0.1:5000/Reinforcement/Cartpole/download_pth')
    const blob = await response.blob();

    // BlobをURLに変換し、ダウンロード用のリンクを作成
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;

    // ダウンロード時のファイル名を指定
    link.setAttribute('download', 'best_CartPole.pth');

    // ダウンロード用のリンクをクリックしてファイルをダウンロード
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handlepyDownload = async () => {
    const response = await fetch('http://127.0.0.1:5000/Reinforcement/Cartpole/download_py')
    const blob = await response.blob();

    // BlobをURLに変換し、ダウンロード用のリンクを作成
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;

    // ダウンロード時のファイル名を指定
    link.setAttribute('download', 'model_config.py');

    // ダウンロード用のリンクをクリックしてファイルをダウンロード
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  return (
    <div>
      <h1>重みファイルのダウンロード</h1>
      <button onClick={handlepthDownload}>.pth</button>
      <h1>pythonファイルのダウンロード</h1>
      <button onClick={handlepyDownload}>.py</button>
    </div>
  );
}

export default CartPoleDownload;
