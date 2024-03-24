

function Getdata(structures) {
  console.log('要素取得', structures);
  const { inputneuron: [inputNeuron] } = structures;
  const { middleneuron: [middleList] } = structures;
  const { outputneuron: [outputNeuron] } = structures;
  // 入力層,出力層のニューラルネットワークの構造を取得
  const otherstructureData = {
    Input_size: inputNeuron,
    Output_size: outputNeuron
  };

  // // 中間層のニューラルネットワークの構造を取得
  const structureList = [];
  middleList.forEach((middle, index) => {
    const structureData = {
      Neuron_num: middle.number,
      Activ_func: middle.activation
    };
    structureList.push(structureData);
  })
  // 学習手段の取得
  const TrainInfoElement = document.getElementById('TrainInfo-wrapper');
  const LossElement = TrainInfoElement.querySelector('.Loss-name');
  const OptimizerElement = TrainInfoElement.querySelector('.Optimizer-name');
  const lrElement = TrainInfoElement.querySelector('.lr-num');
  const BatchElement = TrainInfoElement.querySelector('.batch-num');
  const BufferElement = TrainInfoElement.querySelector('.buffer-size');
  const ActionElement = TrainInfoElement.querySelector('.action-size');
  const EpsilonElement = TrainInfoElement.querySelector('.epsilon-num');
  const EpochElement = TrainInfoElement.querySelector('.epoch-num');
  const TrainInfoData = {
    Loss: LossElement.textContent,
    Optimizer: OptimizerElement.textContent,
    Learning_rate: lrElement.textContent,
    Batch_num: BatchElement.textContent,
    Buffer_size: BufferElement.textContent,
    Action_size: ActionElement.textContent,
    Epsilon: EpsilonElement.textContent,
    Epoch: EpochElement.textContent
  };

  const AllData = {
    structures: structureList,
    other_structure: otherstructureData,
    train_info: TrainInfoData
  };
  return AllData
}

export default Getdata;
