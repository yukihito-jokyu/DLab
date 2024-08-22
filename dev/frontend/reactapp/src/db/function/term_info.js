import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebase";

export async function fetchTermInfo(infoName) {
    const docRef = doc(db, "term_info", infoName);
    const docSnap = await getDoc(docRef);
    if (docSnap.exists()) {
        return docSnap.data().info;
    } else {
        console.log(`No:${infoName}`);
        return [];
    }
}
