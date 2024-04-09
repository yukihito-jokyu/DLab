function GetFlappyData(trainInfo) {
  console.log('Flappyデータ取得', trainInfo);
  const { inputLayer: [inputSize] } = trainInfo;
  const { convLayer: [convList] } = trainInfo;
  const { middleLayer: [middleList] } = trainInfo;
  const { outputLayer: [outputNeuron] } = trainInfo;
  
  // 入力データと出力データのサイズを取得
  const OtherStructure = {
    Input_size: inputSize,
    Output_size: outputNeuron
  };
  
  // 中間層のデータ取得(畳み込み層)
  const conv_structureList = [];
  convList.forEach((conv, index) => {
    const layer_name = conv.LayerName;
    if (layer_name === 'Conv2d') {
      const ConvStructureData = {
        Layer_name: conv.LayerName,
        In_channel: conv.InChannel,
        Out_channel: conv.OutChannel,
        Kernel_size: conv.KernelSize,
        Stride: conv.Stride,
        Padding: conv.Padding,
        Active_func: conv.ActivFunc
      };
      conv_structureList.push(ConvStructureData);
    }
    if (layer_name === 'MaxPool2d') {
      const ConvStructureData = {
        Layer_name: conv.LayerName,
        Kernel_size: conv.KernelSize,
        Stride: conv.Stride,
        Padding: conv.Padding
      }
      conv_structureList.push(ConvStructureData);
    }
  });
  
  // 中間層のデータ取得(全結合層)
  const structureList = [];
  middleList.forEach((middle, index) => {
    const structureData = {
      Neuron_num: middle.number,
      Activ_func: middle.activation
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