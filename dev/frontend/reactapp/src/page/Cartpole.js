import React, { createContext, useState } from 'react';

import './CartPole.css';
import CartPoleLayer from '../components/CartPole/CartPoleLayer';
import DQNTrainInfo from '../components/utils/DQNTrainInfo';
import CartPoleFrame from '../components/CartPole/CartPoleFrame';
import CartPoleDownload from '../components/CartPole/CartPoleDownload';
import Test from '../components/CartPole/Test';
import TestDnD from '../components/CartPole/TestDnD';

export const InputContext = createContext();

function Cartpole(props) {
  // 入力層
  const [inputNeuron, setInputNeuron] = useState(1);
  // 中間層
  const [middleList, setMiddleList] = useState([]);
  // 出力層
  const [outputNeuron, setOutputNeuron] = useState(1);
  const structures = {
    inputneuron: [inputNeuron, setInputNeuron],
    middleneuron: [middleList, setMiddleList],
    outputneuron: [outputNeuron, setOutputNeuron]
  };

  return (
    <InputContext.Provider value={{ structures }}>
      <h1>CartPole</h1>
      <CartPoleLayer />
      <Test/ >
      <DQNTrainInfo />
      <CartPoleFrame id={props.id} />
      <CartPoleDownload />
      <TestDnD />
    </InputContext.Provider>
  )
};

export default Cartpole;