import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "./firebase";
import { collection, doc, getDoc, getDocs, query, setDoc, updateDoc, where } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';

// googleでサインイン
const signInWithGoogle = (setUserId, setFirstSignIn) => {
  // firebaseを使ってグーグルでサインインする
  signInWithPopup(auth, provider).then(() => {
    // ここでfirebaseにユーザー情報が無かったら保存することにする
    searchUserMail(setUserId, setFirstSignIn);
  }).then( async () => {
    // ログイン中のuserIdを取得し、セットする
    const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email))
    const querySnapshot = await getDocs(q);
    if (!querySnapshot.empty) {
      const id = querySnapshot.docs[0].data().user_id;
      setUserId(id);
      sessionStorage.setItem('userId', JSON.stringify(id));
    }
  });
};

// サインアウト
const handlSignOut = (setUserId) => {
  setUserId('');
  sessionStorage.setItem('userId', JSON.stringify(''));
  auth.signOut();
};

// firebaseにユーザー情報があるか確認。無かったら保存
const searchUserMail = async (setUserId, setFirstSignIn) => {
  const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email));
  const querySnapshot = await getDocs(q);
  if (querySnapshot.empty) {
    setFirstSignIn(true);
    saveData(setUserId);
  };
};

// firebaseにユーザー情報新規保存
const saveData = async (setUserId) => {
  const user_id = uuidv4();
  const mail_address = auth.currentUser.email;
  const user_name = user_id;
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  setUserId(user_id);
  sessionStorage.setItem('userId', JSON.stringify(user_id));
  await setDoc(doc(db, "user", user_id), userData);
};

//firebaseに名前を登録
const registName = async (user_id, name) => {
  const userRef = doc(db, 'user', user_id);
  await updateDoc(userRef, {user_name: name});
};

// user_idから参加projectを取得する方法
const getProject = async (userId) => {
  const q = query(collection(db, "participation_projecs"), where("user_id", "==", userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data()
  }
  return null;
}

// プロジェクト情報を取得
const getProjectInfo = async () => {
  const docRef = doc(db, 'project_info', 'info');
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data()
  } else {
    return null
  }
}

// メールアドレスからuser_idを取得する方法
const getUserId = async (mail_address) => {
  const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const user_id = querySnapshot.docs[0].data().user_id;
    sessionStorage.setItem('userId', JSON.stringify(user_id));
  };
};

// test
const testSetDb = async (user_id, mail_address, user_name) => {
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  await setDoc(doc(db, "user", user_id), userData);
};

export { signInWithGoogle, handlSignOut, testSetDb, registName, getProject, getProjectInfo, getUserId };