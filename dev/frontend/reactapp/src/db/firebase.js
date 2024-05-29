// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import firebasekey from "./firebasekey.json"
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: firebasekey.apiKey,
  authDomain: firebasekey.authDomain,
  projectId: firebasekey.projectId,
  storageBucket: firebasekey.storageBucket,
  messagingSenderId: firebasekey.messagingSenderId,
  appId: firebasekey.appId,
  measurementId: firebasekey.measurementId
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export default db;