import React, { useState, } from 'react'
import './TrainInfo.css'

function TrainInfo() {
  const [selectedLoss, setSelectedLoss] = useState('mse_loss');
  const [selectedOptimizer, setSelectedOptimizer] = useState('SGD');
  const [lr, setlr] = useState('0');
  const [batch, setbatch] = useState('0');
  const [buffer, setbuffer] = useState('10000');
  const [action, setaction] = useState('0');
  const [epsilon, setepsilon] = useState('0,01');
  const [epoch, setepoch] = useState('0');
  // const lr_ref = useRef(0);
  // let lr = lr_ref.current
  // console.log(lr_ref.current);

  const handleLossChange = (e) => {
    setSelectedLoss(e.target.value);
  };

  const handlOptimizerChange = (e) => {
    setSelectedOptimizer(e.target.value);
  };

  const handllrChange = (e) => {
    setlr(e.target.value);
  };

  const handlbatchChange = (e) => {
    setbatch(e.target.value);
  };

  const handlbufferChange = (e) => {
    setbuffer(e.target.value);
  };

  const handlactionChange = (e) => {
    setaction(e.target.value);
  };

  const handlepsilonChange = (e) => {
    setepsilon(e.target.value);
  };

  const handlepochChange = (e) => {
    setepoch(e.target.value);
  };

  return (
    <div id='TrainInfo-wrapper'>
      <div className='Loss'>
        <label htmlFor='Loss-Style'>損失関数：</label>
        <select id='Loss-Style' value={selectedLoss} onChange={handleLossChange}>
          <option value="mse_loss">mse_loss</option>
          <option value="cross_entropy">cross_entropy</option>
          <option value="binary_cross_entropy">binary_cross_entropy</option>
          <option value="nll_loss">nll_loss</option>
          <option value="hinge_embedding_loss">hinge_embedding_loss</option>
        </select>
        <p className='Loss-name'>{selectedLoss}</p>
      </div>
      <div className='Optimizer'>
      <label htmlFor='Optimizer-Style'>勾配降下法：</label>
        <select id='Optimizer-Style' value={selectedOptimizer} onChange={handlOptimizerChange}>
          <option value="SGD">SGD</option>
          <option value="momentum">momentum</option>
          <option value="Adam">Adam</option>
          <option value="Adagrad">Adagrad</option>
          <option value="RMSprop">RMSprop</option>
          <option value="Adadelta">Adadelta</option>
        </select>
        <p className='Optimizer-name'>{selectedOptimizer}</p>
      </div>
      <div className='Learning-rate'>
        <label>学習率：</label>
        <input type='number' min='0.0000000001' onChange={handllrChange} step='0.0000000001' />
        <p className='lr-num'>{lr}</p>
      </div>
      <div className='Batch-num'>
        <label>バッチサイズ：</label>
        <input type='number' min='0' onChange={handlbatchChange} step='1' />
        <p className='batch-num'>{batch}</p>
      </div>
      <div className='Buffer-size'>
        <label>バッファサイズ：</label>
        <input type='number' min='0' onChange={handlbufferChange} />
        <p className='buffer-size'>{buffer}</p>
      </div>
      <div className='Action-size'>
        <label>行動の種類：</label>
        <input type='number' min='0' onChange={handlactionChange} step='1' />
        <p className='action-size'>{action}</p>
      </div>
      <div className='Epsilon'>
        <label>ランダムに行動する割合：</label>
        <input type='number' min='0' onChange={handlepsilonChange} step='0.01' />
        <p className='epsilon-num'>{epsilon}</p>
      </div>
      <div className='Epoch'>
        <label>学習回数：</label>
        <input type='number' min='0' onChange={handlepochChange} step='1' />
        <p className='epoch-num'>{epoch}</p>
      </div>
    </div>
  )
}

export default TrainInfo
