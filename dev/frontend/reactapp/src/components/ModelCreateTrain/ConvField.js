import React from 'react'
import './ModelCreateTrain.css';
import MiddleTileField from './MiddleTileField';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';
import { provider } from '../../db/firebase';

function ConvField({ convLayer, setConvLayer, setNowIndex, handleModal, handleDeleteConvTile, setParameter, setParameterSet, setLayerType, setSelectedIndex }) {
  const handleAddTile = () => {
    setNowIndex(0);
    handleModal();
  };
  const onDragEnd = (result) => {
    if (!result.source || !result.destination) {
      return;
    };
    const { source, destination } = result;
    const copyConvLayer = [...convLayer];
    const [removed] = copyConvLayer.splice(source.index, 1);
    copyConvLayer.splice(destination.index, 0, removed);
    const newConvLayer = copyConvLayer;
    setConvLayer(newConvLayer);
  };
  return (
    <div className='middle-field-wrapper'>
      <div className='tile-add-button-over-wrapper'>
        <div className='tile-add-button-wrapper'>
          <div onClick={(handleAddTile)}>
            <TileAdd className='tile-add-button' />
          </div>
        </div>
      </div>
      <DragDropContext onDragEnd={onDragEnd}>
        <div>
          <Droppable droppableId='0'>
            {(provided) => (
              <div
                ref={provided.innerRef}
                {...provided.droppableProps}
              >
                {convLayer.map((conv, index) => (
                  <Draggable
                    draggableId={conv.id}
                    index={index}
                    key={conv.id}
                  >
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                      >
                        <MiddleTileField
                          key={index}
                          tileName={conv.layer_type}
                          layer={conv}
                          setLayer={setConvLayer}
                          index={index}
                          setNowIndex={setNowIndex}
                          handleModal={handleModal}
                          handleDeleteTile={handleDeleteConvTile}
                          setParameter={setParameter}
                          setParameterSet={setParameterSet}
                          setLayerType={setLayerType}
                          setSelectedIndex={setSelectedIndex}
                        />
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </div>
      </DragDropContext>
      {/* {convLayer.map((conv, index) => (
        <MiddleTileField key={index} tileName={conv.layer_type} layer={convLayer} setLayer={setConvLayer} index={index} setNowIndex={setNowIndex} handleModal={handleModal} handleDeleteTile={handleDeleteConvTile} />
        // console.log(index)
      ))} */}
    </div>
  )
}

export default ConvField;
