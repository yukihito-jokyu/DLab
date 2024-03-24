import React, { useState } from 'react';
import './MiddleLayerStyle.css';

function LayerStyle(props) {
  // 引数
  const index = props.index;
  const setData = props.setData;
  const neurondata = props.neuronData;
  const neuronNumber = neurondata.number;
  const activation = neurondata.activation;
  const id = neurondata.id;

  // 要素を隠す
  const [hide, setHide] = useState(true);

  const handleNeuronChange = (e) => {
    setData(prevData => {
      const updatedData = [...prevData];
      const index = updatedData.findIndex(item => item.id === id);
      if (index !== -1) {
        updatedData[index].number = parseInt(e.target.value, 10);
      }
      return updatedData;
    })
  };

  const handleActivationChange = (e) => {
    setData(prevData => {
      const updatedData = [...prevData];
      const index = updatedData.findIndex(item => item.id === id);
      if (index !== -1) {
        updatedData[index].activation = e.target.value;
      }
      return updatedData;
    })
  };

  // クリックされたら要素を消す
  const handledelet = () => {
    props.DeletIvent();
  };

  // クリックしたら要素を隠す
  const handlehide = () => {
    if (hide) {
      setHide(false);
    } else {
      setHide(true);
    }
  }

  return (
    <div className='layer-wrapper'>
      <p>{index}</p>
      <button onClick={handlehide}>縮小・拡大</button>
      {hide && <div>
        <button onClick={handledelet}>-</button>
        <div className='neuron'>
          <label htmlFor="LayerStyle">ニューロン数：</label>
          <select id="LayerStyle" value={neuronNumber} onChange={handleNeuronChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
          <p className='neuron_num'>{neuronNumber}</p>
        </div>
        <div className='activ-func'>
          <label htmlFor="activFuncSelector">活性化関数：</label>
          <select id="activFuncSelector" value={activation} onChange={handleActivationChange}>
            <option value="ReLU">ReLU</option>
            <option value="Sigmoid">Sigmoid</option>
            <option value="Tanh">Tanh</option>
            <option value="Softmax">Softmax</option>
            <option value="None">None</option>
          </select>
          <p className='neuron_activ'>{activation}</p>
        </div>
      </div>}
    </div>
  );
}

export default LayerStyle;
