import { Line } from "react-chartjs-2";
import './ModelManegementEvaluation.css';

const DisplayAcc = ({ accuracyData }) => {
  const optionsAccuracy = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Accuracy Over Epochs',
        font: {
          size: 22,
          weight: 'bold',
        },
      },
      legend: {
        display: true,
        position: 'bottom',
        align: 'end',
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Epoch',
          font: {
            size: 14,
          },
        },
      },
      y: {
        title: {
          display: true,
          text: 'Accuracy',
          font: {
            size: 14,
          },
        },
      },
    },
  };

  return (
    <div className='model-accuracy-picture canvas-container'>
      <Line data={accuracyData} options={optionsAccuracy} />
    </div>
  );
}

export default DisplayAcc;
