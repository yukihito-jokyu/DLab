import { useEffect, useState } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '../db/firebase';

const useFetchTrainingResults = (userId, projectName, modelId) => {
  const [accuracyData, setAccuracyData] = useState(null);
  const [lossData, setLossData] = useState(null);

  useEffect(() => {
    const docRef = doc(db, "training_results", `${userId}_${projectName}_${modelId}`);

    const unsubscribe = onSnapshot(docRef, (doc) => {
      if (doc.exists()) {
        const results = doc.data().results;

        const epochs = results.map(result => result.Epoch);
        const trainAcc = results.map(result => result.TrainAcc);
        const valAcc = results.map(result => result.ValAcc);
        const trainLoss = results.map(result => result.TrainLoss);
        const valLoss = results.map(result => result.ValLoss);

        setAccuracyData({
          labels: epochs,
          datasets: [
            {
              label: 'Train Accuracy',
              data: trainAcc,
              borderColor: 'rgba(54,162,235,1)',
              fill: false,
            },
            {
              label: 'Validation Accuracy',
              data: valAcc,
              borderColor: 'rgba(255,99,132,1)',
              fill: false,
            },
          ],
        });

        setLossData({
          labels: epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: trainLoss,
              borderColor: 'rgba(54,162,235,1)',
              fill: false,
            },
            {
              label: 'Validation Loss',
              data: valLoss,
              borderColor: 'rgba(255,99,132,1)',
              fill: false,
            },
          ],
        });
      }
    });

    return () => unsubscribe();
  }, [userId, projectName, modelId]);

  return { accuracyData, lossData };
};

export default useFetchTrainingResults;
