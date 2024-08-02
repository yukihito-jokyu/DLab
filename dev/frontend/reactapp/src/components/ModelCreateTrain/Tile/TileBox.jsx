import React, { useEffect, useState } from 'react';
import './TileBox.css';
import { ReactComponent as EjectIcon } from '../../../assets/svg/eject_24.svg';
import { ReactComponent as Clip } from '../../../assets/svg/attach_file_24.svg'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';
import Tile from './Tile';

function TileBox({ droppableId, snapshot }) {
  const [dragType, setDragType] = useState('')
  if (snapshot) {
    console.log(snapshot.isDragging)
    console.log(dragType)
  }
  
  const [open, setOpen] = useState(false);
  const [boxLayer, setBoxLayer] = useState([0]);
  const handleClick = () => {
    setOpen(!open);
  };
  const style1 = {
    borderBottom: open ? '3px solid white' : 'none'
  }
  const style2 = {
    height: open ? '500px' : '35px',
    transition: 'all 0.3s'
  }
  const style3 = {
    transform: open ? 'none' : 'rotate(180deg)',
    transition: 'all 0.3s'
  }
  const style4 = {
    height: open ? '400px' : '0px'
  }
  const className = open ? 'tile-box-wrapper opened' : 'tile-box-wrapper'

  useEffect(() => {
    if (snapshot) {
      if (snapshot.idDragging) {
        // setDragType('TILE');
      } else {
        setDragType('');
      }
    }
    
  }, [snapshot]);
  return (
    <div>
      <div className={className}>
        <div className='tile-box-up' style={style1}>
          <p className='tile-title'>Tile Box</p>
          <div className='box-info-wrapper'>
            <div className='tile-num-wrapper'>
              <Clip className='clip-icon' />
              <p>x</p>
            </div>
            <p>xxx,xxx,xxx</p>
          </div>
        </div>
      </div>
      <div className='tile-box' style={style2}>
        <div className='tile-box-field'>
          <div className='box-field' style={style4}>
            {/* ここにdnd実装 */}
            {open && (
              <Droppable
                droppableId={String(droppableId)}
                // type='ITEM'
              >
                {(provided) => (
                  <div
                    ref={provided.innerRef}
                    {...provided.droppableProps}
                    className='droppable-container'
                    style={style4}
                  >
                    {boxLayer.map((box, index) => (
                      <Draggable
                        draggableId={`box2-${index}`}
                        index={index}
                        key={index}
                      >
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                          >
                            <Tile />
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </div>
                )}
              </Droppable>)}
          </div>
        </div>
        <div className='tile-box-open-button' onClick={handleClick}>
          <EjectIcon className='eject-icon' style={style3} />
        </div>
      </div>
    </div>
  )
}

export default TileBox
