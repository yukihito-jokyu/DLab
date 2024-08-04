import { collection, doc, getDoc, getDocs, query, where } from "firebase/firestore";
import { db } from "../firebase";

// 全ての画像分類プロジェクト情報を取得
const getClassificationProjectInfo = async () => {
  const docRef = doc(db, 'project_info', 'image_classification_info');
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data()
  } else {
    return null
  }
}

// 全ての強化学習プロジェクト情報を取得
const getReinforcementlearningProjectInfo = async () => {
  const docRef = doc(db, 'project_info', 'reinforcement_learning_info');
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data()
  } else {
    return null
  }
}

// プロジェクトごとの詳細情報を取得
const getProjectDetailedInfo = async (projectId) => {
  const q = query(collection(db, 'project_info'), where("name", "==", projectId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data();
  } else {
    return null
  }
}

export { getClassificationProjectInfo, getReinforcementlearningProjectInfo, getProjectDetailedInfo }