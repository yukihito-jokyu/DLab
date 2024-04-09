import React, { createContext, useState } from 'react'
import FlappyBirdLayer from '../components/FlappyBird/FlappyBirdLayer'

export const CNNContext = createContext();

function Flappybird(props) {
  // 入力層
  const [inputSize, setInputSize] = useState([28, 28, 1]);
  // 畳み込み層
  const [convList, setConvList] = useState([]);
  // 全結合層
  const [middleList, setMiddleList] = useState([]);
  // 出力層
  const [outputNeuron, setOutputNeuron] = useState(1);
  // 学習の詳細情報
  const [batchSize, setBatchSize] = useState(32);
  const [criterion, setCriterion] = useState('cross_entropy');
  const [optimizer, setOptimizer] = useState('Adam');
  const [epoch, setEpoch] = useState(100);
  const [lr, setLr] = useState(0.01);
  // Context Value
  const trainInfo = {
    inputLayer: [inputSize, setInputSize],
    convLayer: [convList, setConvList],
    middleLayer: [middleList, setMiddleList],
    outputLayer: [outputNeuron, setOutputNeuron],
    info: {
      batchInfo: [batchSize, setBatchSize],
      criterionInfo: [criterion, setCriterion],
      optimizerInfo: [optimizer, setOptimizer],
      epochInfo: [epoch, setEpoch],
      lrInfo: [lr, setLr]
    }
  };
  return (
    <div>
      <CNNContext.Provider value={{ trainInfo }} >
        <h1>Flappybird</h1>
        <FlappyBirdLayer id={props.id} />
      </CNNContext.Provider>
    </div>
  )
}

export default Flappybird
