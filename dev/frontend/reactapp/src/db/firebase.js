// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDAKhMcpRqD3B3vVvNECvx4IxJl7vNadHk",
  authDomain: "dlab-7f511.firebaseapp.com",
  projectId: "dlab-7f511",
  storageBucket: "dlab-7f511.appspot.com",
  messagingSenderId: "279443795048",
  appId: "1:279443795048:web:2c0e0c78c01290c43b1fdd",
  measurementId: "G-VLC07VVZ52"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export default db;