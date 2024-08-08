import { deleteObject, getDownloadURL, getStorage, listAll, ref } from "firebase/storage"
import JSZip from "jszip";
import { saveAs } from "file-saver";
import { storage } from "../firebase"


// 学習曲線の取得
const getImage = async (path) => {
  const imageRef = ref(storage, path);
  try {
    const url = await getDownloadURL(imageRef);
    return url;
  } catch (error) {
    console.log('Error getting image URL:', error);
    return null;
  }
};

// ディレクトリ内の全てのファイル名を取得
const listFilesInDirectory = async (path) => {
  const directoryRef = ref(getStorage(), path);
  try {
    const result = await listAll(directoryRef);
    return result.items; // Array of file references
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

// モデル情報をzipファイルとしてダウンロード
const downloadDirectoryAsZip = async (directoryPath) => {
  const directoryRef = ref(storage, directoryPath);
  const zip = new JSZip();

  try {
    // ディレクトリ内の全ファイルを取得
    const result = await listAll(directoryRef);
    const files = result.items;

    // 各ファイルのURLを取得し、ZIPに追加
    const filePromises = files.map(async (fileRef) => {
      const url = await getDownloadURL(fileRef);
      const response = await fetch(url);
      const blob = await response.blob();
      zip.file(fileRef.name, blob);
    });

    await Promise.all(filePromises);

    // ZIPファイルを生成し、ダウンロード
    const zipBlob = await zip.generateAsync({ type: "blob" });
    saveAs(zipBlob, "archive.zip");
  } catch (error) {
    console.error("Error downloading or zipping files:", error);
  }
};

export { getImage, deleteFilesInDirectory, downloadDirectoryAsZip }