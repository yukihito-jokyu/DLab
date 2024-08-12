import { useState, useEffect } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '../db/firebase';

const useFetchAccuracyAndLoss = (modelId) => {
    const [accuracy, setAccuracy] = useState(null);
    const [loss, setLoss] = useState(null);

    useEffect(() => {
        const docRef = doc(db, 'model_management', modelId);

        const unsubscribe = onSnapshot(docRef, (doc) => {
            if (doc.exists()) {
                const data = doc.data();
                setAccuracy(data.accuracy);
                setLoss(data.loss);
            }
        });

        return () => unsubscribe();
    }, [modelId]);

    return { accuracy, loss };
};

export default useFetchAccuracyAndLoss;
