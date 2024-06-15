import React, { useCallback, useContext } from 'react';
import './RLProjectList.css';
import CartPoleIcon from './CartPoleIcon';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';
import FlappyBirdIcon from './FlappyBirdIcon';
import RLRequestIcon from './RLRequestIcon';
import { useNavigate } from 'react-router-dom';
import { UserInfoContext } from '../../App';

function RLProjectField() {
  const { setProjectId } = useContext(UserInfoContext);
  const style1 = {
    marginRight: '20px'
  };
  const navigate = useNavigate();

  // const handlenav = () => {
  //   navigate('/ModelManegementEvaluation');
  // };
  const handleNav = useCallback((projectId) => {
    navigate('/ModelManegementEvaluation')
    sessionStorage.setItem('projectId', JSON.stringify(projectId));
  }, [navigate])

  return (
    <div className='rl-project-field-wrapper'>
      <div className='field-middle'>
        <div onClick={() => handleNav("CartPole")}>
          <BorderGradationBox style1={style1}>
            <CartPoleIcon />
          </BorderGradationBox>
        </div>
        <div onClick={() => handleNav("FlappyBird")}>
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
