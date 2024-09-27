import React, { useContext, useEffect, useState } from 'react';
import { db, auth } from './firebase';
import { collection, doc, getDocs, query, where } from 'firebase/firestore';
import { getProjectInfo, handlSignOut, signInWithGoogle, testSetDb } from './firebaseFunction';
import { useAuthState } from "react-firebase-hooks/auth";
import { UserInfoContext } from '../App';
import { useNavigate } from 'react-router-dom';

import { socket } from '../socket/socket';


function TestFirebase() {
  const [posts, setPosts] = useState([]);
  const [user] = useAuthState(auth);
  const { userId, setUserId, setFirstSignIn } = useContext(UserInfoContext);
  const navigate = useNavigate();

  // ソケット通信のてすと
  const [data, setData] = useState(null);

  // useEffect(() => {
  //   const postData = collection(db, 'user');
  //   getDocs(postData).then((snapShot) => {
  //     // console.log(snapShot.docs.map((doc) => ({...doc.data()})));
  //     setPosts(snapShot.docs.map((doc) => ({...doc.data()})))
  //   })
  // }, []);

  const getData = async () => {
    const userSnapshot = await getDocs(collection(db, 'user'));
    // userSnapshot.forEach((doc) => {
    //   console.log(doc.data().mail_address);
    // });
    const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email))
    console.log(q.docs == null)
    const querySnapshot = await getDocs(q);
    console.log(querySnapshot.docs[0].data())
    querySnapshot.forEach((doc) => {
      console.log(doc.data())
    })
    // console.log(querySnapshot.docs[0].data())
  }

  const checkUserId = () => {
    console.log(userId);
  };

  // ソケット通信のテスト
  const handleStartSocket = () => {
    console.log("起動")
    socket.emit('test', {})
  }

  useEffect(() => {
    const socketFire = (data) => {
      console.log(data);
      setData(data.response);
    }
    
    socket.on("test_event", socketFire);
    
    return () => {
      socket.off("test_event", socketFire);
    };
  }, [])

  // ストリーミング
  const handleStartStreem = () => {
    const eventSource = new EventSource('http://localhost:5000/stream');
    eventSource.onmessage = (event) => {
      console.log("発火")
      console.log(event);
      setData(event);
    };
  }

  return (
    <div>
      <div>
        {posts.map((post) => (
          <div key={post.mail_address}>
            <div>{post.user_id}</div>
            <p>{post.user_name}</p>
            <p>{post.mail_address}</p>
            {/* <p>{post.timestanp}</p> */}
          </div>
        ))}
      </div>
      <button onClick={() => signInWithGoogle(setUserId, setFirstSignIn)}><p>サインイン</p></button>
      <button onClick={() => handlSignOut(setUserId)}><p>サインアウト</p></button>
      {user && <div>
        <p>{auth.currentUser.displayName}</p>
      </div>}
      <button onClick={testSetDb}><p>データ追加</p></button>
      {/* <p>{user}</p> */}
      <button onClick={getData}><p>データ取得</p></button>
      <button onClick={checkUserId}><p>ユーザーid取得</p></button>
      <button onClick={() => navigate('/top')}>ページ遷移</button>
      <button onClick={getProjectInfo}>プロジェクト情報取得</button>
      <div>
        <button onClick={handleStartSocket}>ソケット通信</button>
      </div>
      <div>
        {data}
      </div>
      <div>
        <button onClick={handleStartStreem}>ストリーミング</button>
      </div>
      <div>
        {data}
      </div>
    </div>
  )
}

export default TestFirebase
