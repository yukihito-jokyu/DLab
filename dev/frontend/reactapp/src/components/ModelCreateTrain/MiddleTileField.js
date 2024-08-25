import React from 'react'
import './ModelCreateTrain.css';
import NuronTile from './Tile/NuronTile';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'
import Conv2dTile from './Tile/Conv2dTile';
import Tile from './Tile/Tile';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import TileBox from './Tile/TileBox';

function MiddleTileField({ tileName, layer, index, setNowIndex, handleModal, handleDeleteTile, setParameter, setLayerType, setSelectedIndex, shape, snapshot }) {
  const handleAddTile = () => {
    setNowIndex(index + 1);
    handleModal();
  };
  const handleParameter = () => {
    setParameter(layer);
    setLayerType(layer.type);
    setSelectedIndex(index);
  };
  return (
    <div className='over-wrapper'>
      <div className='middle-tile-field-wrapper'>
        <div className='delete-tile'>
          <div className='delete-button-wrapper'>
            <div className='delete-button' onClick={() => handleDeleteTile(index)} style={{ cursor: 'pointer' }}>
              <DeletIcon className='delet-svg' />
            </div>
          </div>
        </div>
        <div className='tile-position-wrapper' onClick={handleParameter} style={{ cursor: 'pointer' }}>
          {tileName === 'Conv2d' ? (
            <Conv2dTile shape={shape} />
          ) : tileName === 'Linear' ? (
            <NuronTile shape={shape} />
          ) : tileName === 'Box' ? (
            <TileBox droppableId={layer.box_id} snapshot={snapshot} />
          ) : (
            <Tile text={tileName} shape={shape} />
          )}
          <div className='tile-add-button-wrapper'>
            <div onClick={(handleAddTile)} style={{ cursor: 'pointer' }}>
              <TileAdd className='tile-add-button' />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MiddleTileField
