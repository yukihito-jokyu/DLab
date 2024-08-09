import React from 'react'
import './ModelCreateTrain.css'
import GradationFonts from '../../uiParts/component/GradationFonts'
import Tile from './Tile/Tile'
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';

function TileAddModal({ handleModal, handleAddTile }) {
  const text = 'タイルを選択してください'
  const style = {
    fontSize: "26px",
    fontWeight: "600"
  }
  const layerList = ['Conv2d', 'MaxPool2d', 'BatchNorm', 'Dropout', 'Linear']
  return (
    <div>
      <div className='tile-add-modal-wrapper'></div>
      <div className='tile-add-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='tile-add-field'>
              <div className='modal-title'>
                <GradationFonts text={text} style={style} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='modal-tile-field'>
                {layerList.map((layer, index) => (
                  <div key={index} className='modal-tile-wrapper' onClick={() => handleAddTile(layer)}>
                    <Tile text={layer} />
                  </div>
                ))}
              </div>
              <div className='layer-add-modal-delet-button-field' onClick={handleModal}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
      
    </div>
  )
}

export default TileAddModal
