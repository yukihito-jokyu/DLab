import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "../firebase";
import { arrayUnion, collection, doc, getDocs, query, setDoc, updateDoc, where } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';

// googleアカウントでサインイン
const signInWithGoogle = (setFirstSignIn) => {
  // firebaseを使ってグーグルでサインインする
  signInWithPopup(auth, provider).then(() => {
    // ここでfirebaseにユーザー情報が無かったら保存することにする
    searchUserMailAddress(setFirstSignIn);
  }).then( async () => {
    // ログイン中のuserIdを取得し、セットする
    const q = query(collection(db, "users"), where("mail_address", "==", auth.currentUser.email))
    const querySnapshot = await getDocs(q);
    if (!querySnapshot.empty) {
      const id = querySnapshot.docs[0].data().user_id;
      // setUserId(id);
      sessionStorage.setItem('userId', JSON.stringify(id));
    }
  });
};

// firebaseにユーザー情報があるか確認。無かったら保存
const searchUserMailAddress = async (setFirstSignIn) => {
  const q = query(collection(db, "users"), where("mail_address", "==", auth.currentUser.email));
  const querySnapshot = await getDocs(q);
  if (querySnapshot.empty) {
    setFirstSignIn(true);
    initUsers();
  };
};

// firebaseにユーザー情報新規保存
const initUsers = async () => {
  const user_id = uuidv4();
  const mail_address = auth.currentUser.email;
  const user_name = user_id;
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name,
    favorite_user: [],
    join_project: []
  };
  const sentData = {
    user_id: user_id
  };
  const response = await fetch('http://127.0.0.1:5000/mkdir/user', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(sentData),
  });
  console.log(response);
  // setUserId(user_id);
  sessionStorage.setItem('userId', JSON.stringify(user_id));
  await setDoc(doc(db, "users", user_id), userData);
};

// user_nameの登録
const registName = async (user_id, name) => {
  const userRef = doc(db, 'users', user_id);
  await updateDoc(userRef, {user_name: name});
};

// サインアウト
const handlSignOut = () => {
  sessionStorage.setItem('userId', JSON.stringify(''));
  auth.signOut();
};

// メールアドレスからuserIdを取得する
const getUserId = async (mail_address) => {
  const q = query(collection(db, "users"), where("mail_address", "==", auth.currentUser.email));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const user_id = querySnapshot.docs[0].data().user_id;
    sessionStorage.setItem('userId', JSON.stringify(user_id));
  };
};

// 画像分類参加プロジェクトの取得
const getJoinProject = async (userId) => {
  const q = query(collection(db, "users"), where("user_id", "==", userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data()['join_project'];
  }
  return [];
};

// 画像分類の参加プロジェクトの更新
const updateJoinProject = async (userId, projectName) => {
  const q = query(collection(db, "users"), where("user_id", "==", userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const doc = querySnapshot.docs[0]
    await updateDoc(doc.ref, {
      join_project: arrayUnion(projectName)
    });
  }
};

// userIdからuserNameを取得する
const getUserName = async (userId) => {
  const q = query(collection(db, 'users'), where('user_id', '==', userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().user_name;
  } else {
    return null
  }
};

// お気に入りユーザーを取得
const getFavoriteUser = async (userId) => {
  const q = query(collection(db, 'users'), where('user_id', '==', userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().favorite_user;
  } else {
    return []
  }
}

export { signInWithGoogle, registName, handlSignOut, getUserId, getJoinProject, updateJoinProject, getUserName, getFavoriteUser }