import React, { useContext } from 'react'
import { InputContext } from '../../page/Cartpole'

function Test() {
  const { inputData } = useContext(InputContext);
  return (
    <div>
      <p>入力ニューロン数：{inputData}</p>
    </div>
  )
}

export default Test
