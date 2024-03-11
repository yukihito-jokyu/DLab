import React from 'react'
import FlappyInputLayer from './FlappyInputLayer'
import FlappyConvLayer from './FlappyConvLayer'
import MiddleLayer from '../CartPole/MiddleLayer'
import InputOutputLayerStyle from '../CartPole/InputOutputLayerStyle'
import DQNTrainInfo from '../utils/DQNTrainInfo'
import FlappyFrame from './FlappyFrame'

function FlappyBirdLayer() {
  return (
    <div>
      <FlappyInputLayer />
      <FlappyConvLayer />
      <MiddleLayer />
      <h1>出力層</h1>
      <div id='output_size'>
        <InputOutputLayerStyle />
      </div>
      <DQNTrainInfo />
      <FlappyFrame />
    </div>
  )
}

export default FlappyBirdLayer
