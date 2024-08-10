import React, { useState } from 'react'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd'
import { v4 as uuidv4 } from 'uuid';

// css
import './DnDFild.css'

function CNNDnDFild(props) {
  // コンポーネートの受け取り
  const MiddleLayerStyle = props.middleLayer;
  const [convList, setConvList] = props.middleData;

  const [data, setData] = useState([]);

  // test
  let input_num = [160, 160, 1];
  const [inputList, setInputList] = useState([[160, 160, 1]]);


  const onDragEnd = (result) => {
    if (!result.source || !result.destination) {
      // ドラッグが不適切な場合など、source や destination が存在しない場合は何もせずに終了
      return;
    }

    const { source, destination } = result;


    console.log(convList);
    const sourceNuron = [...convList];
    // タスクの削除
    const [removed] = sourceNuron.splice(source.index, 1);
    // タスクの追加
    sourceNuron.splice(destination.index, 0, removed);

    const newdata = sourceNuron;
    setConvList(newdata);
  };

  // 要素の追加
  const handleAdd = () => {
    const newNuron = {
      id: uuidv4(),
      LayerName: "Conv2d",
      InChannel: 3,
      OutChannel: 64,
      KernelSize: 3,
      Stride: 1,
      Padding: 0,
      ActivFunc: "ReLU"
    };
    setConvList(prevData => [...prevData, newNuron]);
  };
  // 要素の削除
  const handleDeletion = (idToDelete) => {
    setConvList(prevData => prevData.filter(nuron => nuron.id !== idToDelete));
  };



  // html追加
  const handleMakeHTML = (id, index) => {
    const { value, element } = MiddleLayerStyle({ DeletIvent: () => handleDeletion(id), Index: index });
    console.log('受け取った値：', value);
    // console.log(inputList);
    // setInputList([...inputList, value]);
    input_num = value;
    return element
  }
  return (
    <div className='dnd-fild'>
      <button onClick={handleAdd} style={{ cursor: 'pointer' }}>+</button>
      <DragDropContext onDragEnd={onDragEnd}>
        <div className="trello">
          <Droppable droppableId='0'>
            {(provided) => (
              <div
                className='nuron-box'
                ref={provided.innerRef}
                {...provided.droppableProps}
              >
                <div className='nuron-data'>
                  {convList.map((nuron_data, index) => (
                    <Draggable
                      draggableId={nuron_data.id}
                      index={index}
                      key={nuron_data.id}
                    >
                      {(provided, snapshot) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                          style={{
                            ...provided.draggableProps.style,
                            border: snapshot.isDragging ? "2px solid red" : "0.5px solid #1c0909"
                          }}
                        >
                          <MiddleLayerStyle DeletIvent={() => handleDeletion(nuron_data.id)} setData={setConvList} neuronData={nuron_data} id={nuron_data.id} />
                          {/* {handleMakeHTML(nuron_data.id, index)}
                          {console.log(index)} */}
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              </div>
            )}
          </Droppable>
        </div>
      </DragDropContext>
    </div>
  );
}

export default CNNDnDFild;
