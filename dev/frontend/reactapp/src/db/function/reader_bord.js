import { collection, getDocs, orderBy, query } from "firebase/firestore";
import { db } from "../firebase";

// リーダーボードの情報を取得
const getReaderBoardInfo = async (projectId) => {
  let projectName;
  projectName = projectId + "_reader_board";
  const q = await query(collection(db, projectName), orderBy('accuracy', 'desc'));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs;
  } else {
    return []
  }
};

// ユーザーidから順位を取得する方法
const getUserRank = async (projectId, userId) => {
  const collectionName = `${projectId}_reader_board`;
  const q = await query(collection(db, collectionName), orderBy('accuracy', 'desc'));
  const querySnapshot = await getDocs(q);
  let rank = 'NaN';
  console.log(userId)
  querySnapshot.docs.forEach((doc, index) => {
    console.log(doc.data().user_id)
    if (doc.data().user_id === userId) {
      rank = index + 1; // 1から始まる順位
    }
  });
  return rank
}

export { getReaderBoardInfo, getUserRank }