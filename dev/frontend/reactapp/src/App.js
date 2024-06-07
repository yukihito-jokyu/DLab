// Reactアプリのコード (frontend/src/App.jsなど)
import React, { useState } from 'react';
// import Layer from "./components/CartPole/CartPoleLayer";
// import DQNTrainInfo from './components/utils/DQNTrainInfo';
// import CSVUploader from './components/CSVUploader';
// import CartPoleFrame from './components/CartPole/CartPoleFrame';
// ページ遷移用
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import './App.css'


// 関数
// import Getdata from './utils/Getdata';
// import SocketContext from '.';

import Test from './page/Test';
// import FlappyBird from './components/FlappyBird';
// import Dnd from './components/Dnd';
import Home from './page/Home';
import Reinforcement from './page/Reinforcement'
import ImageRecognition from './page/ImageRecognition';
import Flappybird from './page/Flappybird';
import CartPole from './page/Cartpole';
import ImageClassificationProjectList from './components/ImageClassificationProjectList/ImageClassificationProjectList';
import ModelManegementEvaluation from './components/ModelManegementEvaluation/ModelManegementEvaluation';
import ModelCreateTrain from './components/ModelCreateTrain/ModelCreateTrain';
import RLProjectList from './pages/component/RLProjectList';
import Login from './components/Login/Login';
import TestFirebase from './db/TestFirebase';
import DjangoTest from './Django/DjangoTest';
import Top from './components/Top/Top';
import { UserIdContext } from './context/context';

function App() {
  const [userId, setUserId] = useState("");
  // ソケット通信
  // const socket = useContext(SocketContext);

  
  // csvファイル読み込み
  // const handleCSVUpload = (csvData) => {
  //   console.log('CSVデータ：', csvData);
  // };
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
  // const [count, setCount] = useState(0);
  
  // const ref = useRef();
  // const handlClick = async () => {
  //   const startTime = new Date();
  //   const data = {'count': count};
  //   const response = await fetch('http://127.0.0.1:5000/', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify(data),
  //   });
  //   const responseData = await response.json();
  //   console.log('Flaskからの応答:', responseData.count);
  //   setCount(responseData.count);
    
  //   // setCount(count + 1);
  //   const endTime = new Date();
  //   const elapsedTime = endTime - startTime;
  //   console.log("処理にかかった時間: " + elapsedTime + "ミリ秒");
  //   console.log(count);
  // };

  // useEffect(() => {
  //   console.log('Hello Hooks');
  // }, [count]);

  // const getStructure = async (e) => {
  //   const AllData = Getdata();
  //   const response = await fetch('http://127.0.0.1:5000/train', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify(AllData),
  //   });
  //   console.log('受信')
  //   console.log(response)
  // };

  // const handlRef = () => {
  //   console.log(ref.current.value);
  // };

  // idをリストで管理する。
  const [elementId, setElementId] = useState(['0']);

  // クリックしたときにリストを追加する。
  const handleMakeId = () => {
    setElementId(prev => [...prev, `${prev.length+1}`]);
  };

  return (
    <UserIdContext.Provider value={[userId, setUserId]}>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path='/Login' element={<Login />} />
          <Route path="/test" element={<Test />} />
          <Route path='/Reinforcement' element={<Reinforcement handlemakeid={handleMakeId} elementid={elementId} />} />
          <Route path='/ImageRecognition' element={<ImageRecognition />} />
          {elementId.map((id, index) => (
            <Route key={index} path={'/Reinforcement/Cartpole'+id} element={<CartPole id={id} />} />
          ))}
          {elementId.map((id, index) => (
            <Route key={index} path={'/Reinforcement/Flappybird'+id} element={<Flappybird id={id} />} />
          ))}
          <Route path='/top' element={<Top />} />
          <Route path="/ImageClassificationProjectList" element={<ImageClassificationProjectList />} />
          <Route path="/ModelManegementEvaluation" element={<ModelManegementEvaluation />} />
          <Route path="/ModelCreateTrain" element={<ModelCreateTrain />} />
          <Route path="/RLProjectList" element={<RLProjectList />} />
          <Route path="/testfirebase" element={<TestFirebase />} />
          <Route path="/testdjango" element={<DjangoTest />} />
        </Routes>
      </Router>
      {/* <div>-</div>
      <button onClick={handlClick}>+</button>
      <p>{count}</p>
      <h1>useContext</h1>
      <h1>useRef</h1>
      <input type='text' ref={ref} />
      <button onClick={handlRef}>sent</button>
      <h1>CSVファイル読み込み</h1>
      <CSVUploader onCSVUpload={handleCSVUpload} /> */}
      {/* <h1>クリックと同時に要素追加</h1>
      <button onClick={getStructure}>要素を取得</button>
      <Layer />
      <h1>学習の手段</h1>
      <DQNTrainInfo /> */}
      {/* <div>
        <button onClick={handleSocket}>データ処理を開始</button>
        <p>進歩：{progress}</p>
        <p>受信データ：{data}</p>
      </div>
      <div>
        <button onClick={handldisconnect}>Socket切断</button>
      </div> */}
      {/* <CartPoleFrame />
      <FlappyBird />
      <Dnd /> */}
    </UserIdContext.Provider>
  );
}

export default App;