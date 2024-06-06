import React, { useEffect, useState } from 'react';
import { db, auth } from './firebase';
import { collection, getDocs } from 'firebase/firestore';
import { signInWithGoogle, testSetDb } from './firebaseFunction';
import { useAuthState } from "react-firebase-hooks/auth";
import { signOut } from 'firebase/auth';


function TestFirebase() {
  const [posts, setPosts] = useState([]);
  const [user] = useAuthState(auth);
  console.log(user);

  useEffect(() => {
    const postData = collection(db, 'user');
    getDocs(postData).then((snapShot) => {
      // console.log(snapShot.docs.map((doc) => ({...doc.data()})));
      setPosts(snapShot.docs.map((doc) => ({...doc.data()})))
    })
  }, []);

  


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
      <button onClick={signInWithGoogle}><p>サインイン</p></button>
      <button onClick={() => auth.signOut()}><p>サインアウト</p></button>
      {user && <div>
        <p>{auth.currentUser.displayName}</p>
      </div>}
      <button onClick={testSetDb}><p>データ追加</p></button>
      {/* <p>{user}</p> */}
    </div>
  )
}

export default TestFirebase
