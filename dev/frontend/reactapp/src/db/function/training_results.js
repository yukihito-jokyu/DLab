import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebase";

export async function fetchTrainingResults(userId, projectName, modelId) {
    const docRef = doc(db, "training_results", `${userId}_${projectName}_${modelId}`);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
        return docSnap.data().results;
    } else {
        console.log(`No:${userId}_${projectName}_${modelId}`);
        return [];
    }
}