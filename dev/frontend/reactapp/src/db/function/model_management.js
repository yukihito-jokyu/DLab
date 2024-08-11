import { collection, deleteDoc, doc, getDocs, getDoc, query, serverTimestamp, setDoc, where } from "firebase/firestore";
import { db } from "../firebase";
import { deleteModelStructure, initModelStructure } from "./model_structure";
import { v4 as uuidv4 } from 'uuid';

// project_idとuser_idを用いてデータベースからmodel_idを取得する方法
const getModelId = async (user_id, project_id) => {
  const q = query(
    collection(db, "model_management"),
    where("user_id", "==", user_id),
    where("project_id", "==", project_id)
  );
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const dataList = [];
    querySnapshot.docs.map((doc) => (
      dataList.push({ id: doc.id, ...doc.data() })
    ));
    return dataList
  };
  return []
};

// モデルを新規作成
const initModel = async (userId, projectId, modelId, modelName) => {
  const accuracy = null;
  const loss = null;
  const date = serverTimestamp();
  const modelData = {
    accuracy: accuracy,
    date: date,
    loss: loss,
    model_id: modelId,
    model_name: modelName,
    project_id: projectId,
    user_id: userId,
    status: 'pre'
  };
  await setDoc(doc(db, "model_management", modelId), modelData);
  await initModelStructure(modelId, projectId);
};

// モデル情報の削除
const deleteModels = async (modelId) => {
  // モデル情報を削除
  const q = query(collection(db, 'model_management'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (doc) => {
    await deleteDoc(doc.ref);
  });

  // 学習結果のドキュメントを削除
  const trainingDocRef = doc(db, 'training_results', modelId);
  const trainingDocSnapshot = await getDoc(trainingDocRef);
  if (trainingDocSnapshot.exists()) {
    await deleteDoc(trainingDocRef);
  }

  // モデル構造の削除
  await deleteModelStructure(modelId);
};


// modelIdからmodelNameを取得する
const getModelName = async (modelId) => {
  const q = query(collection(db, 'model_management'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().model_name;
  } else {
    return null
  }
};

// project_idとuser_idを用いてデータベースからmodel_nameを取得する方法
const getModelNames = async (user_id, project_id) => {
  const q = query(
    collection(db, "model_management"),
    where("user_id", "==", user_id),
    where("project_id", "==", project_id)
  );
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const dataList = [];
    querySnapshot.docs.map((doc) => (
      dataList.push({ model_name: doc.model_name, ...doc.data() })
    ));
    return dataList
  };
  return []
};

export { getModelId, initModel, deleteModels, getModelName, getModelNames }