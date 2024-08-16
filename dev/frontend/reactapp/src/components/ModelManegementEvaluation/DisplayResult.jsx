import { Line } from "react-chartjs-2";
import './ModelManegementEvaluation.css';

const DisplayResult = ({ data, type, showTitle = true }) => {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: showTitle,
                text: `${type} Over Epochs`,
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
                    text: type,
                    font: {
                        size: 14,
                    },
                },
            },
        },
    };

    return (
        <div className={`model-accuracy-picture canvas-container`}>
            <Line data={data} options={options} />
        </div>
    );
}

export default DisplayResult;
