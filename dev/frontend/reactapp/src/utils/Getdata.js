

function Getdata() {
  console.log('要素取得')
  // 入力層,出力層のニューラルネットワークの構造を取得
  const InputStructureElement = document.getElementById('input');
  const InputSize = InputStructureElement.querySelector('.neuron_num');
  const OutputStructureElement = document.getElementById('output');
  const OutputSize = OutputStructureElement.querySelector('.neuron_num');
  const otherstructureData = {
    Input_size: InputSize.textContent,
    Output_size: OutputSize.textContent
  };
  console.log(otherstructureData)

  // 中間層のニューラルネットワークの構造を取得
  const structureList = [];
  const MiddleStructureElement = document.getElementById('structure');
  Array.from(MiddleStructureElement.children).forEach((element) => {
    const neuronNumElement = element.querySelector('.neuron_num');
    const neuronActivElement = element.querySelector('.neuron_activ');
    const structureData = {
      Neuron_num: neuronNumElement.textContent,
      Activ_func: neuronActivElement.textContent
    };
    structureList.push(structureData);
  });

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
  console.log(AllData);
  return AllData
}

export default Getdata
