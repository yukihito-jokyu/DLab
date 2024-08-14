import { collection, deleteDoc, doc, getDocs, query, setDoc, updateDoc, where } from "firebase/firestore";
import { db } from "../firebase";

// モデルの構造の初期化
const initModelStructure = async (modelId, projectId) => {
  let shape = 24
  let originShape = 24
  let C = 1
  if (projectId === 'CIFAR10') {
    shape = 32
    originShape = 32
    C = 3
  }
  let newData
  if (projectId === 'CartPole') {
    newData = {
      model_id: modelId,
      train_info: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        optimizer: "Adam",
        buffer: 10000,
        episilon: 0.1,
        syns: 20
      },
      structure: {
        input_layer: {
          shape: 4,
          type: "Input"
        },
        middle_layer: [],
        output_layer: 2
      }
    }
  } else if (projectId === 'FlappyBird') {
    newData = {
      model_id: modelId,
      origin_shape: 32,
      train_info: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        optimizer: "Adam",
        buffer: 10000,
        episilon: 0.1,
        syns: 20
      },
      structure: {
        input_layer: {
          shape: [32, 32, 3],
          change_shape: 32,
          preprocessing: "none",
          type: "Input"
        },
        conv_layer: [],
        flatten_method: {
          type: "Flatten",
          way: "normal"
        },
        middle_layer: [],
        output_layer: 2
      }
    }
  } else {
    newData = {
      model_id: modelId,
      origin_shape: originShape,
      train_info: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        optimizer: "Adam",
        test_size: 0.2,
        image_shape: shape
      },
      structure: {
        input_layer: {
          shape: [shape, shape, C],
          change_shape: shape,
          preprocessing: "none",
          type: "Input"
        },
        conv_layer: [],
        flatten_method: {
          type: "Flatten",
          way: "normal"
        },
        middle_layer: [],
        output_layer: 10
      }
    }
  }
  await setDoc(doc(db, "model_structure", modelId), newData);
};

// モデルの構造の削除
const deleteModelStructure = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (doc) => {
    await deleteDoc(doc.ref);
  });
};

// モデルの構造データの取得
const getModelStructure = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().structure;
  } else {
    return null
  }
};

// モデルの訓練情報の取得
const getTrainInfo = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().train_info;
  } else {
    return null
  }
};

// モデルの画像サイズの取得
const getImageShape = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().structure.input_layer.change_shape;
  } else {
    return null
  }
};

// モデルの画像サイズの取得
const getPreprocessing = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().structure.input_layer.preprocessing;
  } else {
    return null
  }
};

// モデルの構造の更新
const updateStructure = async (modelId, structure) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (document) => {
    const docRef = doc(db, 'model_structure', document.id);
    await updateDoc(docRef, {
      structure: structure
    });
  })
};

// モデルの訓練情報の保存
const updateTrainInfo = async (modelId, trainInfo) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (document) => {
    const docRef = doc(db, 'model_structure', document.id);
    await updateDoc(docRef, {
      train_info: trainInfo
    });
  })
};

// モデルのオリジン画像サイズを取得
const getOriginShape = async (modelId) => {
  const q = query(collection(db, 'model_structure'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().origin_shape;
  } else {
    return null
  }
};

export { initModelStructure, deleteModelStructure, getModelStructure, getTrainInfo, updateStructure, updateTrainInfo, getOriginShape, getImageShape, getPreprocessing }