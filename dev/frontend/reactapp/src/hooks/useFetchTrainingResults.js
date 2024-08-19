import { useEffect, useState } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '../db/firebase';

const useFetchTrainingResults = (modelId, task) => {
  const [accuracyData, setAccuracyData] = useState({ labels: [], datasets: [] });
  const [lossData, setLossData] = useState({ labels: [], datasets: [] });
  const [totalRewardData, setTotalRewardData] = useState({ labels: [], datasets: [] });
  const [averageLossData, setAverageLossData] = useState({ labels: [], datasets: [] });

  useEffect(() => {
    const docRef = doc(db, "training_results", modelId);

    const unsubscribe = onSnapshot(docRef, (doc) => {
      if (doc.exists()) {
        const data = doc.data();

        const options = {
          responsive: true,
          maintainAspectRatio: false,
          elements: {
            point: {
              radius: 4,
              hoverRadius: 7,
              backgroundColor: '#fbfbfb',
            },
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: function (tooltipItem) {
                  const datasetLabel = tooltipItem.dataset.label || '';
                  const value = tooltipItem.raw || '';
                  return `${datasetLabel}: ${value}`;
                },
              },
            },
          },
        };

        if (task === 'ImageClassification') {
          const results = data.results || [];

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
                ...options,
              },
              {
                label: 'Validation Accuracy',
                data: valAcc,
                borderColor: 'rgba(255,99,132,1)',
                fill: false,
                ...options,
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
                ...options,
              },
              {
                label: 'Validation Loss',
                data: valLoss,
                borderColor: 'rgba(255,99,132,1)',
                fill: false,
                ...options,
              },
            ],
          });

        } else if (task === 'ReinforcementLearning') {
          const results = data.results || [];

          const epochs = results.map(result => result.Epoch);
          const totalReward = results.map(result => result.TotalReward);
          const avgLoss = results.map(result => result.AverageLoss);

          setTotalRewardData({
            labels: epochs,
            datasets: [
              {
                label: 'Total Reward',
                data: totalReward,
                borderColor: 'rgba(54,162,235,1)',
                fill: false,
                ...options,
              },
            ],
          });

          setAverageLossData({
            labels: epochs,
            datasets: [
              {
                label: 'Average Loss',
                data: avgLoss,
                borderColor: 'rgba(255,99,132,1)',
                fill: false,
                ...options,
              },
            ],
          });
        }
      }
    });

    return () => unsubscribe();
  }, [modelId, task]);

  return { accuracyData, lossData, totalRewardData, averageLossData };
};

export default useFetchTrainingResults;
