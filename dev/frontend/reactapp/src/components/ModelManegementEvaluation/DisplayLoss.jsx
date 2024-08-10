import { Line } from "react-chartjs-2";
import './ModelManegementEvaluation.css';

const DisplayLoss = ({ lossData }) => {
  const optionsLoss = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Loss Over Epochs',
        font: {
          size: 22,
          weight: 'bold',
        },
      },
      legend: {
        display: true,
        position: 'top',
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
          text: 'Loss',
          font: {
            size: 14,
          },
        },
      },
    },
  };

  return (
    <div className='model-loss-picture canvas-container'>
      <Line data={lossData} options={optionsLoss} />
    </div>
  )
}

export default DisplayLoss;
