import React from 'react'
import FlappyInputLayer from './FlappyInputLayer'
import FlappyConvLayer from './FlappyConvLayer'
import DQNTrainInfo from '../utils/DQNTrainInfo'
import FlappyFrame from './FlappyFrame'
import FlappyBirdMiddleLayer from './FlappyBirdMiddleLayer'
import FlappyBirdOutputLayerStyle from './FlappyBirdOutputLayerStyle'

function FlappyBirdLayer(props) {
  return (
    <div>
      <FlappyInputLayer />
      <FlappyConvLayer />
      <FlappyBirdMiddleLayer />
      <h1>出力層</h1>
      <FlappyBirdOutputLayerStyle />
      <DQNTrainInfo />
      <FlappyFrame id={props.id} />
    </div>
  )
}

export default FlappyBirdLayer
