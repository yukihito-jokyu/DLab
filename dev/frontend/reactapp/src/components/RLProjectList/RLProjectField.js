import React, { useCallback, useContext } from 'react';
import './RLProjectList.css';
import CartPoleIcon from './CartPoleIcon';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';
import FlappyBirdIcon from './FlappyBirdIcon';
import RLRequestIcon from './RLRequestIcon';
import { useNavigate } from 'react-router-dom';
import { UserInfoContext } from '../../App';

function RLProjectField() {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const style1 = {
    marginRight: '20px'
  };
  const navigate = useNavigate();

  // const handlenav = () => {
  //   navigate('/ModelManegementEvaluation');
  // };
  const handleNav = useCallback((projectId) => {
    navigate(`/ModelManegementEvaluation/${userId}/${projectId}`)
    sessionStorage.setItem('projectId', JSON.stringify(projectId));
  }, [navigate, userId])

  return (
    <div className='rl-project-field-wrapper'>
      <div className='field-middle'>
        <div onClick={() => handleNav("CartPole")} style={{ cursor: 'pointer' }}>
          <BorderGradationBox style1={style1}>
            <CartPoleIcon />
          </BorderGradationBox>
        </div>
        <div onClick={() => handleNav("FlappyBird")} style={{ cursor: 'pointer' }}>
          <BorderGradationBox style1={style1}>
            <FlappyBirdIcon />
          </BorderGradationBox>
        </div>
        <RLRequestIcon />
      </div>
    </div>
  );
}

export default RLProjectField;
