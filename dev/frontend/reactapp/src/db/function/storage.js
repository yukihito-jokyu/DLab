import { deleteObject, getDownloadURL, getStorage, listAll, ref, uploadBytes } from "firebase/storage"
import JSZip from "jszip";
import { saveAs } from "file-saver";
import { storage } from "../firebase"


const getImage = async (path) => {
  const imagePath = await listFilesInDirectory(path);
  console.log(imagePath.length);
  if (imagePath.length !== 0) { // ここを修正
    const imageRef = ref(storage, `${path}/${imagePath[0].name}`); // ここも修正
    try {
      const url = await getDownloadURL(imageRef);
      return url;
    } catch (error) {
      console.log('Error getting image URL:', error);
      return null;
    }
  }
  return null;
};

const listFilesInDirectory = async (path) => {
  const directoryRef = ref(getStorage(), path);
  try {
    const result = await listAll(directoryRef);
    console.log("Files in directory:", result.items);
    return result.items;
  } catch (error) {
    console.error("Error listing files:", error);
    return [];
  }
};

// ディレクトリ内の全てのファイルの削除
const deleteFilesInDirectory = async (path) => {
  const files = await listFilesInDirectory(path);
  try {
    await Promise.all(files.map(fileRef => deleteObject(fileRef)));
    console.log("All files deleted successfully");
  } catch (error) {
    console.error("Error deleting files:", error);
  }
};

// ユーザーの画像をアップロード
const uploadUserImage = async (userId, imageFile, imageType) => {
  await deleteFilesInDirectory(`images/${userId}`)
  try {
    const storageRef = ref(storage, `images/${userId}/profile.${imageType}`);
    await uploadBytes(storageRef, imageFile);
    const downloadURL = await getDownloadURL(storageRef);
    console.log('File uploaded successfully. Download URL:', downloadURL);
  } catch (error) {
    console.error('Error uploading file:', error);
  }
}

// 個別のZIPファイルを作成する関数
const createZipFromDirectory = async (directoryPath) => {
  const storage = getStorage();
  const directoryRef = ref(storage, directoryPath);

  const zip = new JSZip();

  try {
    const result = await listAll(directoryRef);
    const downloadPromises = result.items.map(async (itemRef) => {
      const url = await getDownloadURL(itemRef);
      const response = await fetch(url);
      const blob = await response.blob();
      zip.file(itemRef.name, blob);
    });

    await Promise.all(downloadPromises);

    // ZIPファイルを生成してBlobとして返す
    const content = await zip.generateAsync({ type: 'blob' });
    return content;
  } catch (error) {
    console.error('Error creating ZIP from directory:', error);
    throw error;
  }
};

// 複数のZIPファイルをまとめる関数
const combineZipsIntoOne = async (childZipBlobs, models) => {
  const parentZip = new JSZip();

  try {
    // 各子ZIPファイルを親ZIPファイルに追加
    for (let i = 0; i < childZipBlobs.length; i++) {
      const childZipBlob = childZipBlobs[i];
      parentZip.file(`${models[i].model_name}.zip`, childZipBlob);
    }

    // 親ZIPファイルを生成してBlobとして返す
    const content = await parentZip.generateAsync({ type: 'blob' });
    return content;
  } catch (error) {
    console.error('Error creating parent ZIP file:', error);
    throw error;
  }
};


export { getImage, deleteFilesInDirectory, createZipFromDirectory, combineZipsIntoOne, uploadUserImage }