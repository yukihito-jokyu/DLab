// Reactアプリのコード (frontend/src/App.jsなど)
import React, { useEffect, useState, useRef } from 'react';
import Socket from 'socket.io-client';
import Layer from "./components/Layer";
import TrainInfo from './components/TrainInfo';
import CSVUploader from './components/CSVUploader';

// 関数
import Getdata from './utils/Getdata';

function App() {
  // ソケット通信
  const socket = Socket('ws://127.0.0.1:5000');
  socket.on('end', (data) => {
    console.log(data);
  })

  const handleTrain = () => {
    const AllData = Getdata();
    console.log(AllData);
    socket.emit('CartPole', AllData);
  };
  // csvファイル読み込み
  const handleCSVUpload = (csvData) => {
    console.log('CSVデータ：', csvData);
  };
  // test
  // const [progress, setprogress] = useState(0);
  // const [data, setdata] = useState('');
  // const [socket, setSocket] = useState(null);

  // useEffect(() => {
  //   console.log('socketの立ち上げ');
  //   const newsocket = Socket('ws://127.0.0.1:5000');
  //   // newsocket.disconnect();
  //   setSocket(newsocket);
  
  //   // サーバーからのメッセージを受信
  //   newsocket.on('update', (data) => {
  //     setprogress(data.progress);
  //     setdata(data.data);
  //   });
  
  //   // 処理が完了したらソケットを閉じる
  //   const handleComplete = (data) => {
  //     console.log(data.message);
  //     console.log(newsocket.connected)
  //     // newsocket.disconnect();
      
  //     // newsocket.close(); // 強制的にクローズ
  //     console.log(newsocket.connected)
  //   };
  
  //   newsocket.on('complete', handleComplete);
  
  //   return () => {
  //   };
  // }, []);

  // const handleSocket = () => {
  //   // サーバーにデータ処理の開始リクエストを送信
  //   // このリクエストに対する途中結果や最終結果がWebSocketを通じて受信される
  //   if (socket) {
  //     console.log('起動');
  //     socket.connect();
  //     console.log(socket.connected)
      
  //     socket.emit('start_process', {});
  //   } else {
  //     console.error('Socketが初期化されません');
  //   }
  // };

  // const handldisconnect = () => {
  //   console.log('ソケットを閉じる')
  //   socket.disconnect();
  //   console.log(socket.connected)
  // };

  // const [data, setData] = useState([]);

  // useEffect(() => {
  //   const fetchData = async () => {
  //     console.log('test1');
  //     try {
  //       const response = await fetch('http://127.0.0.1:5000/');  // Flaskアプリのエンドポイントに合わせてURLを設定
  //       console.log(response);
  //       const result = await response.json();
  //       console.log(result);
  //       setData(result);
  //     } catch (error) {
  //       console.error('Error fetching data:', error);
  //     }
  //   };
  //   fetchData();
  // }, []);
  const [count, setCount] = useState(0);
  
  const ref = useRef();
  const handlClick = async () => {
    const startTime = new Date();
    const data = {'count': count};
    const response = await fetch('http://127.0.0.1:5000/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    const responseData = await response.json();
    console.log('Flaskからの応答:', responseData.count);
    setCount(responseData.count);
    
    // setCount(count + 1);
    const endTime = new Date();
    const elapsedTime = endTime - startTime;
    console.log("処理にかかった時間: " + elapsedTime + "ミリ秒");
    console.log(count);
  };

  useEffect(() => {
    console.log('Hello Hooks');
  }, [count]);

  const getStructure = async (e) => {
    const AllData = Getdata();
    const response = await fetch('http://127.0.0.1:5000/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(AllData),
    });
    console.log('受信')
    console.log(response)
  };

  const handlRef = () => {
    console.log(ref.current.value);
  };

  return (
    <div>
      <h1>Data from Flask:</h1>
      {/* <ul>
        {data.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul> */}
      <button onClick={handlClick}>+</button>
      <p>{count}</p>
      <h1>useContext</h1>
      <h1>useRef</h1>
      <input type='text' ref={ref} />
      <button onClick={handlRef}>sent</button>
      <h1>CSVファイル読み込み</h1>
      <CSVUploader onCSVUpload={handleCSVUpload} />
      <h1>クリックと同時に要素追加</h1>
      <button onClick={getStructure}>要素を取得</button>
      <Layer />
      <h1>学習の手段</h1>
      <TrainInfo />
      {/* <div>
        <button onClick={handleSocket}>データ処理を開始</button>
        <p>進歩：{progress}</p>
        <p>受信データ：{data}</p>
      </div>
      <div>
        <button onClick={handldisconnect}>Socket切断</button>
      </div> */}
      <button onClick={handleTrain}>学習開始</button>
    </div>
  );
}

export default App;