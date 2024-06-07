import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "./firebase";
import { collection, doc, query, setDoc, where } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';
import { useContext } from "react";
import { UserIdContext } from "../context/context";

// googleでサインイン
const signInWithGoogle = () => {
  // firebaseを使ってグーグルでサインインする
  signInWithPopup(auth, provider).then(() => {
    // ここでfirebaseにユーザー情報が無かったら保存することにする
    searchUserMail()
  }).then(() => {
    // ログイン中のuserIdを取得し、セットする
    const [userId, setUserId] = useContext(UserIdContext);
    const q = query(collection(db, "user"), where("mail_address", "==", auth.currentUser.email))
    const id = q.docs[0].data().user_id;
    setUserId(id);
  });
};

// サインアウト
const signOut = () => {
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
  if (q.docs == null) {
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

export { signInWithGoogle, signOut, testSetDb };