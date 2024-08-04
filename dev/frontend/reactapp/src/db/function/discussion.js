import { arrayUnion, collection, doc, getDoc, getDocs, orderBy, query, serverTimestamp, setDoc, updateDoc } from "firebase/firestore";
import { v4 as uuidv4 } from 'uuid';
import { db } from "../firebase";

// 記事を投稿
const postArticle = async (projectId, userId, userName, title, comment) => {
  const projectName = projectId + '_discussion'
  const date = serverTimestamp();
  const articleId = uuidv4();
  const articleData = {
    user_name: userName,
    user_id: userId,
    title: title,
    date: date,
    comments: [{
      comment: comment,
      user_id: userId,
      user_name: userName
    }]
  }
  await setDoc(doc(db, projectName, articleId), articleData);
};

// ディスカッションのタイトルを取得
const getDiscussionTitle = async (projectId, commentId) => {
  const projectName = projectId + '_discussion';
  const docRef = doc(db, projectName, commentId);
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data().title;
  } else {
    return []
  }
};

// ディスカッションのコメントを取得
const getDiscussionComment = async (projectId, commentId) => {
  const projectName = projectId + '_discussion';
  const docRef = doc(db, projectName, commentId);
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data().comments;
  } else {
    return []
  }
};

// ディスカッションのコメントを追加
const addDiscussionComment = async (projectId, commentId, comment, userId, userName) => {
  const projectName = projectId + '_discussion';
  const articleRef = doc(db, projectName, commentId);
  const docSnap = await getDoc(articleRef);
  const newComment = {
    comment: comment,
    comment_id: docSnap.data().comments.length+1,
    user_id: userId,
    user_name: userName
  };
  await updateDoc(articleRef, {
    comments: arrayUnion(newComment)
  });
};

// ディスカッションの全てのタイトル名の取得
const getDiscussionInfos = async (projectName) => {
  const projectId = projectName + "_discussion";
  const q = await query(collection(db, projectId), orderBy('date', 'desc'));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs;
  } else {
    return null
  }
};

export { postArticle, getDiscussionTitle, getDiscussionComment, addDiscussionComment, getDiscussionInfos }