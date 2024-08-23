import React, { useState, useEffect } from 'react';
import './InformationModal.css';
import GradationFonts from '../../../uiParts/component/GradationFonts';
import { ReactComponent as DeletIcon } from '../../../assets/svg/delet_40.svg';
import { fetchTermInfo } from '../../../db/function/term_info';
import { PropagateLoader } from 'react-spinners';

function InformationModal({ infoName, handleDelete }) {
  const [termInfo, setTermInfo] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      const data = await fetchTermInfo(infoName);
      setTermInfo(data);
      setLoading(false);
    }
    fetchData();
  }, [infoName]);

  const fontStyle1 = {
    fontSize: '26px',
    fontWeight: '600'
  };

  return (
    <div>
      <div className='information-modal-wrapper'></div>
      <div className='tile-add-field-wrapper'>
        {loading ? (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
          }}>
            <PropagateLoader color="linear-gradient(91.27deg, #C49EFF 0.37%, #47A1FF 99.56%)" size={20} />
          </div>
        ) : (
          <div className='gradation-border'>
            <div className='gradation-wrapper'>
              <div className='info-modal-field'>
                <div className='modal-title'>
                  <GradationFonts text={infoName} style={fontStyle1} />
                </div>
                <div className='gradation-border2-wrapper'>
                  <div className='gradation-border2'></div>
                </div>
                <div className='exp-field'>
                  <p>{termInfo}</p>
                </div>
                <div className='train-modal-delet-button-field' onClick={() => handleDelete(false)} style={{ cursor: 'pointer' }}>
                  <DeletIcon className='delet-svg' />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default InformationModal;
