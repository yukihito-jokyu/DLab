import React, { useState } from 'react'
import { DragDropContext, Draggable, Droppable } from "react-beautiful-dnd";
import nuronData from './nuronData';
import MiddleLayerStyle from './CartPole/MiddleLayerStyle';
import './Dnd.css';
import { v4 as uuidv4 } from 'uuid';

function Dnd() {
  const [data, setData] = useState(nuronData);

  const onDragEnd = (result) => {
    if (!result.source || !result.destination) {
      // ドラッグが不適切な場合など、source や destination が存在しない場合は何もせずに終了
      return;
    }
    
    const { source, destination } = result;
    

    console.log(data);
    const sourceNuron = [...data];
    // タスクの削除
    const [removed] = sourceNuron.splice(source.index, 1);
    // タスクの追加
    sourceNuron.splice(destination.index, 0, removed);

    const newdata = sourceNuron;
    setData(newdata);
  };

  // 要素の追加
  const handleAdd = () => {
    const newNuron = {
      id: uuidv4(),
    };
    setData(prevData => [...prevData, newNuron]);
  };
  // 要素の削除
  const handleDeletion = (idToDelete) => {
    setData(prevData => prevData.filter(nuron => nuron.id !== idToDelete));
  };
  return (
    <div>
      <h1>Trello風アプリ</h1>
      <button onClick={handleAdd}>+</button>
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
                  {data.map((nuron_data, index) => (
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
                          <button onClick={() => handleDeletion(nuron_data.id)}>-</button>
                          <MiddleLayerStyle />
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
  )
}

export default Dnd
