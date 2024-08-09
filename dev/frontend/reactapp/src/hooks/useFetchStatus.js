import { useState, useEffect } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '../db/firebase';

const useFetchStatus = (modelId) => {
    const [currentStatus, setCurrentStatus] = useState(null);

    useEffect(() => {
        const docRef = doc(db, 'model_management', modelId);

        const unsubscribe = onSnapshot(docRef, (doc) => {
            if (doc.exists()) {
                const data = doc.data();
                setCurrentStatus(data.status);
            }
        });

        return () => unsubscribe();
    }, [modelId]);

    return currentStatus;
};

export default useFetchStatus;
