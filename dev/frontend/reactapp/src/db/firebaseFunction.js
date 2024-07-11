import { signInWithPopup } from "firebase/auth";
import { auth, db, provider } from "./firebase";
import { arrayUnion, collection, deleteDoc, doc, getDoc, getDocs, orderBy, query, serverTimestamp, setDoc, updateDoc, where } from "firebase/firestore";
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
  const sentData = {
    user_id: user_id
  };
  const response = await fetch('http://127.0.0.1:5000/mkdir/user', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(sentData),
  });
  console.log(response);
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

// プロジェクト情報を取得(上のコードの更新)
const getProjectInfoUp = async (projectId) => {
  const q = query(collection(db, 'project_info'), where("name", "==", projectId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data();
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

// project_idとuser_idを用いてデータベースからmodel_idを取得する方法
const getModelId = async (user_id, project_id) => {
  const q = query(
    collection(db, "models"),
    where("user_id", "==", user_id),
    where("project_id", "==", project_id)
  );
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    const dataList = [];
    querySnapshot.docs.map((doc) => (
      dataList.push({ id: doc.id, ...doc.data() })
    ));

    // console.log(dataList);
    return dataList
  };
  return null
};

// モデルを新規作成
const setModel = async (userId, projectId, modelName) => {
  const modelId = uuidv4();
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
    user_id: userId
  };
  await setDoc(doc(db, "models", modelId), modelData);
  await initModelStructure(modelId, projectId);
}

// モデルの削除
const deleteModels = async (modelId) => {
  const q = query(collection(db, 'models'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (doc) => {
    await deleteDoc(doc.ref);
  });
  await deleteModelStructure(modelId);
};

// Discussionの情報取得
const getDiscussionInfo = async (projectName) => {
  const projectId = projectName + "_discussion";
  const q = await query(collection(db, projectId), orderBy('date', 'desc'));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs;
  } else {
    return null
  }
};

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

// userIdからuserNameを取得する
const getUserName = async (userId) => {
  const q = query(collection(db, 'user'), where('user_id', '==', userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().user_name;
  } else {
    return null
  }
};

// ディスカッションのコメントを追加
const addComment = async (projectId, commentId, comment, userId, userName) => {
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
}

// ディスカッションのコメントだけ取得
const getComment = async (projectId, commentId) => {
  const projectName = projectId + '_discussion';
  const docRef = doc(db, projectName, commentId);
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data().comments;
  } else {
    return null
  }
};

// ディスカッションのタイトルを取得
const getTitle = async (projectId, commentId) => {
  const projectName = projectId + '_discussion';
  const docRef = doc(db, projectName, commentId);
  const docSnap = await getDoc(docRef);
  if (docSnap.exists()) {
    return docSnap.data().title;
  } else {
    return null
  }
};

// リーダーボード
const getReaderBoard = async (projectId) => {
  const projectName = projectId + "_reader_board";
  const q = await query(collection(db, projectName), orderBy('accuracy', 'desc'));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs;
  } else {
    return null
  }
};

// お気に入りユーザーを取得
const getFavoriteUser = async (userId) => {
  const q = query(collection(db, 'user'), where('user_id', '==', userId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().favorite_user;
  } else {
    return null
  }
}

// モデルの構造の初期化
const initModelStructure = async (modelId, projectId) => {
  let newData
  if (projectId === 'CartPole') {
    newData = {
      model_id: modelId,
      TrainInfo: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        loss: "mse_loss",
        optimizer: "Adam",
        buffer: 10000,
        episilon: 0.1,
        syns: 20
      },
      structure: {
        InputLayer: {
          shape: 4,
          type: "Input"
        },
        MiddleLayer: [],
        OutputLayer: 2
      }
    }
  } else if (projectId === 'FlappyBird') {
    newData = {
      model_id: modelId,
      TrainInfo: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        loss: "mse_loss",
        optimizer: "Adam",
        buffer: 10000,
        episilon: 0.1,
        syns: 20
      },
      structure: {
        InputLayer: {
          shape: [28, 28, 1],
          preprocessing: "none",
          type: "Input"
        },
        ConvLayer: [],
        FlattenWay: {
          type: "Flatten",
          way: "normal"
        },
        MiddleLayer: [],
        OutputLayer: 2
      }
    }
  } else {
    newData = {
      model_id: modelId,
      TrainInfo: {
        batch: 32,
        epoch: 100,
        learning_rate: 0.01,
        loss: "mse_loss",
        optimizer: "Adam"
      },
      structure: {
        InputLayer: {
          shape: [28, 28, 1],
          preprocessing: "none",
          type: "Input"
        },
        ConvLayer: [],
        FlattenWay: {
          type: "Flatten",
          way: "normal"
        },
        MiddleLayer: [],
        OutputLayer: 10
      }
    }
  }
  await setDoc(doc(db, "model", modelId), newData);
}

// モデルの構造の削除
const deleteModelStructure = async (modelId) => {
  const q = query(collection(db, 'model'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (doc) => {
    await deleteDoc(doc.ref);
  });
}

// モデルの構造データ受け取り
const getModelStructure = async (modelId) => {
  const q = query(collection(db, 'model'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().structure;
  } else {
    return null
  }
}

// モデルの構造の更新
const updateStructure = async (modelId, structure) => {
  const q = query(collection(db, 'model'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (document) => {
    const docRef = doc(db, 'model', document.id);
    await updateDoc(docRef, {
      structure: structure
    });
  })
}

// モデルの訓練情報の取得
const getTrainInfo = async (modelId) => {
  const q = query(collection(db, 'model'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs[0].data().TrainInfo;
  } else {
    return null
  }
}

// モデルの訓練情報の保存
const updateTrainInfo = async (modelId, trainInfo) => {
  const q = query(collection(db, 'model'), where('model_id', '==', modelId));
  const querySnapshot = await getDocs(q);
  querySnapshot.forEach(async (document) => {
    const docRef = doc(db, 'model', document.id);
    await updateDoc(docRef, {
      TrainInfo: trainInfo
    });
  })
}

// test
const testSetDb = async (user_id, mail_address, user_name) => {
  const userData = {
    mail_address: mail_address,
    user_id: user_id,
    user_name: user_name
  };
  await setDoc(doc(db, "user", user_id), userData);
};

export { signInWithGoogle, handlSignOut, testSetDb, registName, getProject, getProjectInfo, getUserId, getModelId, setModel, deleteModels, getProjectInfoUp, getDiscussionInfo, postArticle, getUserName, addComment, getComment, getTitle, getReaderBoard, getFavoriteUser, getModelStructure, updateStructure, getTrainInfo, updateTrainInfo };