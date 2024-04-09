// 例: Testコンポーネント
import React, { useEffect, useState } from 'react';
import TestSocket from '../components/Tests/TestSocket';

function Test() {
  // idをリストで管理する。
  const [elementId, setElementId] = useState(['0']);
  const [inputData, setInputData] = useState('');
  const [getData, setGetData] = useState('');

  // データの送信
  const handleSendData = async () => {
    const response = await fetch('http://localhost:4000/sendData', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: inputData })
    });
    const result = await response.json();
    console.log(result);
  }

  // データの受信
  const handleGetData = async () => {
    const response = await fetch('http://192.168.0.12:4000/getData', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    const result = await response.json();
    console.log(result.message);
    setGetData(result.message);
  }

  useEffect(() => {
    // データの受信
    const handleGetData = async () => {
      const response = await fetch('http://192.168.0.12:4000/getData', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const result = await response.json();
      console.log(result.message);
      setGetData(result.message);
    }
    // ページを離れた時に実行
    return () => {
      handleGetData();
    };
  }, []);

  // クリックしたときにリストを追加する。
  const handleMakeId = () => {
    setElementId(prev => [...prev, `${prev.length+1}`]);
  };

  // クリックしたときにpygameのwindowを表示する
  const handleOpenPygame = async () => {
    const response = await fetch('http://127.0.0.1:5050/test/pygame', {
      method: 'GET'
    });
    const result = await response.json();
    console.log(result);
  };

  return (
    <div>
      <div>
        <input type='text' value={inputData} onChange={(e) => setInputData(e.target.value)} />
        <button onClick={handleSendData}>Send Data</button>
      </div>
      <div>
        <button onClick={handleGetData}>Get Data</button>
        <p>受信したデータは：{getData}</p>
      </div>
      <div>
        <h1>socket通信テスト</h1>
        <button onClick={handleMakeId}>+</button>
        <div>
          {elementId.map((value, index) => (
            <TestSocket key={index} id={String(index)} />
          ))}
        </div>
      </div>
      <div>
        <h1>pygame</h1>
        <button onClick={handleOpenPygame}>起動</button>
      </div>
    </div>
  );
};

export default Test;

