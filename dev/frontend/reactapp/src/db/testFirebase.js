import React, { useEffect, useState } from 'react';
import db from './firebase';
import { collection, getDocs } from 'firebase/firestore';

function TestFirebase() {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    const postData = collection(db, 'posts');
    getDocs(postData).then((snapShot) => {
      // console.log(snapShot.docs.map((doc) => ({...doc.data()})));
      setPosts(snapShot.docs.map((doc) => ({...doc.data()})))
    })
  }, []);


  return (
    <div>
      <div>
        {posts.map((post) => (
          <div key={post.title}>
            <div>{post.title}</div>
            <p>{post.test}</p>
            {/* <p>{post.timestanp}</p> */}
          </div>
        ))}
      </div>
    </div>
  )
}

export default TestFirebase
