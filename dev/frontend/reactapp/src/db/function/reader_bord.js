import { collection, getDocs, orderBy, query } from "firebase/firestore";
import { db } from "../firebase";

// リーダーボードの情報を取得
const getReaderBoardInfo = async (projectId) => {
  let projectName
  if (projectId === 'CIFAR10') {
    projectName = 'CIFAR-10_reader_board';
  } else {
    projectName = projectId + "_reader_board";
  }
  const q = await query(collection(db, projectName), orderBy('accuracy', 'desc'));
  const querySnapshot = await getDocs(q);
  if (!querySnapshot.empty) {
    return querySnapshot.docs;
  } else {
    return []
  }
};

export { getReaderBoardInfo }