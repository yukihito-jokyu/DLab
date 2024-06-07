import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "./firebase";
import { collection, doc, getDocs, query, setDoc, where } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';

// googleでサインイン
const signInWithGoogle = (setUserId) => {
  // firebaseを使ってグーグルでサインインする
  signInWithPopup(auth, provider).then(() => {
    // ここでfirebaseにユーザー情報が無かったら保存することにする
    searchUserMail()
  }).then( async () => {
    // ログイン中のuserIdを取得し、セットする
    const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email))
    const querySnapshot = await getDocs(q);
    const id = querySnapshot.docs[0].data().user_id;
    setUserId(id);
  });
};

// サインアウト
const handlSignOut = (setUserId) => {
  setUserId('');
  auth.signOut();
};

// firebaseにユーザー情報新規保存
const saveData = async () => {
  const user_id = uuidv4();
  const mail_address = auth.currentUser.email;
  const user_name = user_id;
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  await setDoc(doc(db, "user", user_id), userData);
}

// firebaseにユーザー情報があるか確認。無かったら保存
const searchUserMail = async () => {
  const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email))
  const querySnapshot = await getDocs(q);
  if (querySnapshot.empty) {
    saveData();
  };
}

// test
const testSetDb = async (user_id, mail_address, user_name) => {
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  await setDoc(doc(db, "user", user_id), userData);
}

export { signInWithGoogle, handlSignOut, testSetDb };