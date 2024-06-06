import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "./firebase";
import { doc, setDoc } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';


const signInWithGoogle = () => {
  // firebaseを使ってグーグルでサインインする
  signInWithPopup(auth, provider).then(() => {
    saveData();
  });
};

const signOut = () => {
  auth.signOut();
};

const saveData = async () => {
  const user_id = uuidv4();
  const mail_address = auth.currentUser.email;
  const user_name = 'test';
  await testSetDb(user_id, mail_address, user_name);
}

const testSetDb = async (user_id, mail_address, user_name) => {
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  await setDoc(doc(db, "user", user_id), userData);
}

export { signInWithGoogle, signOut, testSetDb };