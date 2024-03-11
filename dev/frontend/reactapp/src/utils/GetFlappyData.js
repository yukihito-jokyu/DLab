function GetFlappyData() {
  console.log('Flappyデータ取得');
  // 入力データと出力データのサイズを取得
  const InputSize = document.querySelector('.Input_size').textContent.match(/\d+/g);
  const OutputSize = document.getElementById('output_size').querySelector('.neuron_num').textContent;
  const OtherStructure = {
    Input_size: InputSize,
    Output_size: OutputSize
  };
  // 中間層のデータ取得(畳み込み層)
  const conv_structureList = [];
  const ConvStructureElement = document.getElementById('Conv-structure').querySelector('.nuron-data');
  Array.from(ConvStructureElement.children).forEach((element) => {
    const layer_name = element.querySelector('.Layer_name').textContent;
    if (layer_name === 'Conv2d') {
      const ConvStructureData = {
        Layer_name: layer_name,
        In_channel: element.querySelector('.In_channel').textContent,
        Out_channel: element.querySelector('.Out_channel').textContent,
        Kernel_size: element.querySelector('.Kernel_size').textContent,
        Stride: element.querySelector('.Stride').textContent,
        Padding: element.querySelector('.Padding').textContent,
        Active_func: element.querySelector('.Active_func').textContent
      }
      conv_structureList.push(ConvStructureData);
    }
    if (layer_name === 'MaxPool2d') {
      const ConvStructureData = {
        Layer_name: layer_name,
        Kernel_size: element.querySelector('.Kernel_size').textContent,
        Stride: element.querySelector('.Stride').textContent,
        Padding: element.querySelector('.Padding').textContent
      }
      conv_structureList.push(ConvStructureData);
    }
  });
  // 中間層のデータ取得(全結合層)
  const structureList = [];
  const StructureElement = document.getElementById('structure').querySelector('.nuron-data');
  Array.from(StructureElement.children).forEach((element) => {
    const neuronNumElement = element.querySelector('.neuron_num');
    const neuronActivElement = element.querySelector('.neuron_activ');
    const structureData = {
      Neuron_num: neuronNumElement.textContent,
      Activ_func: neuronActivElement.textContent
    };
    structureList.push(structureData);
  });
  const Structure = {
    Conv_Layer: conv_structureList,
    Fully_Connected_Layer: structureList
  };

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
    other_structure: OtherStructure,
    Structure: Structure,
    train_info: TrainInfoData
  };
  return AllData;
}

export default GetFlappyData;