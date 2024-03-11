import React from 'react';

import './CartPole.css';
import CartPoleLayer from '../components/CartPole/CartPoleLayer';
import DQNTrainInfo from '../components/utils/DQNTrainInfo';
import CartPoleFrame from '../components/CartPole/CartPoleFrame';
import CartPoleDownload from '../components/CartPole/CartPoleDownload';

function Cartpole() {
  return (
    <div>
      <h1>CartPole</h1>
      <CartPoleLayer />
      <DQNTrainInfo />
      <CartPoleFrame />
      <CartPoleDownload />
    </div>
  )
};

export default Cartpole;
