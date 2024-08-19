// Reactアプリのコード (frontend/src/App.jsなど)
import React, { createContext, useState } from 'react';
// import Layer from "./components/CartPole/CartPoleLayer";
// import DQNTrainInfo from './components/utils/DQNTrainInfo';
// import CSVUploader from './components/CSVUploader';
// import CartPoleFrame from './components/CartPole/CartPoleFrame';
// ページ遷移用
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

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
import RLProjectList from './components/RLProjectList/RLProjectList';
// import Login from './components/Login/Login';
import TestFirebase from './db/TestFirebase';
import DjangoTest from './Django/DjangoTest';
import Top from './components/Top/Top';
import ProjectShare from './components/ProjectShare/ProjectShare';
import Community from './components/Community/Community';
import Profile from './components/Profile/Profile';
import { socket } from './socket/socket';


export const UserInfoContext = createContext()

function App() {
  const [userId, setUserId] = useState('');
  const [projectId, setProjectId] = useState("");
  const [firstSignIn, setFirstSignIn] = useState(false);
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
    setElementId(prev => [...prev, `${prev.length + 1}`]);
  };

  return (
    <UserInfoContext.Provider value={{ userId, setUserId, firstSignIn, setFirstSignIn, projectId, setProjectId }}>
      <Router>
        <Routes>
          {/* <Route path="/" element={<Home />} /> */}
          {/* <Route path='/Login' element={<Login />} /> */}
          <Route path="/test" element={<Test />} />
          <Route path='/Reinforcement' element={<Reinforcement handlemakeid={handleMakeId} elementid={elementId} />} />
          <Route path='/ImageRecognition' element={<ImageRecognition />} />
          {elementId.map((id, index) => (
            <Route key={index} path={'/Reinforcement/Cartpole' + id} element={<CartPole id={id} />} />
          ))}
          {elementId.map((id, index) => (
            <Route key={index} path={'/Reinforcement/Flappybird' + id} element={<Flappybird id={id} />} />
          ))}
          <Route path='/top' element={<Top />} />
          <Route path="/ImageClassificationProjectList/:userId" element={<ImageClassificationProjectList />} />
          <Route path="/ModelManegementEvaluation/:userId/:task/:projectName" element={<ModelManegementEvaluation />} />
          <Route path="/ModelCreateTrain/:task/:projectName/:modelId" element={<ModelCreateTrain />} />
          <Route path="/RLProjectList" element={<RLProjectList />} />
          <Route path="/testfirebase" element={<TestFirebase />} />
          <Route path="/testdjango" element={<DjangoTest />} />
          <Route path='/projectshare' element={<ProjectShare />} />
          <Route path='/community/:projectName' element={<Community />} />
          <Route path='/profile/:profileUserId' element={<Profile />} />
          <Route path="*" element={<Navigate to="/top" replace />} />
        </Routes>
      </Router>
    </UserInfoContext.Provider>
  );
}

export default App;