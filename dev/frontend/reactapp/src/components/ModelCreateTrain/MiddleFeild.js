import React from 'react';
import './ModelCreateTrain.css';
import MiddleTileField from './MiddleTileField';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';

function MiddleFeild({ middleLayer, setMiddleLayer, setNowIndex, handleModal, handleDeleteMiddleTile, setParameter, setParameterSet, setLayerType, setSelectedIndex }) {
  const handleAddTile = () => {
    setNowIndex(0);
    handleModal();
  };
  const onDragEnd = (result) => {
    if (!result.source || !result.destination) {
      return;
    };
    const { source, destination } = result;
    const copyMiddleLayer = [...middleLayer];
    const [removed] = copyMiddleLayer.splice(source.index, 1);
    copyMiddleLayer.splice(destination.index, 0, removed);
    const newMiddleLayer = copyMiddleLayer;
    setMiddleLayer(newMiddleLayer);
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
          <Droppable droppableId='1'>
            {(provided) => (
              <div
                ref={provided.innerRef}
                {...provided.droppableProps}
              >
                {middleLayer.map((middle, index) => (
                  <Draggable
                    draggableId={middle.id}
                    index={index}
                    key={middle.id}
                  >
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                      >
                        <MiddleTileField
                          key={index}
                          tileName={middle.layer_type}
                          layer={middle}
                          setLayer={setMiddleLayer}
                          index={index}
                          setNowIndex={setNowIndex}
                          handleModal={handleModal}
                          handleDeleteTile={handleDeleteMiddleTile}
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
      {/* {middleLayer.map((middle, index) => (
        <MiddleTileField key={index} tileName={middle.layer_type} layer={middleLayer} setLayer={setMiddleLayer} index={index} setNowIndex={setNowIndex} handleModal={handleModal} handleDeleteTile={handleDeleteMiddleTile} />
      ))} */}
    </div>
  )
}

export default MiddleFeild
