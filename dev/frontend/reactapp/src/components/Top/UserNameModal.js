import React, { useContext, useState } from 'react';
import { UserInfoContext } from '../../App';
import { registName, searchUserName } from '../../db/function/users';
import './Top.css';
import GradationButton from '../../uiParts/component/GradationButton';
import GradationFonts from '../../uiParts/component/GradationFonts';

function UserNameModal({ setFirstSignIn, setSameName }) {
  const [name, setName] = useState("");
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  // const { userId, setFirstSignIn } = useContext(UserInfoContext);
  const handleSetName = async () => {
    const sameName = await searchUserName(name);
    if (sameName) {
      setSameName(true);
    } else {
      setFirstSignIn(false);
      await registName(userId, name)
    }
  }

  const style = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  };

  const handleChange = (e) => {
    setName(e.target.value);
  };

  return (
    <div>
      <div className='alert-modal-wrapper'></div>
      <div className='alert-modal-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='set-username-box-border'>
              <div className='set-username-field'>
                <div className='set-username-wapper'>
                  <div className='user-name'>
                    {/* <p>Project Name</p> */}
                    <GradationFonts text={'ユーザー名'} style={style} />
                  </div>
                  <div className='project-name-field'>
                    {/* <p>Project Name</p> */}
                    <input type='text' placeholder='ユーザー名を記入' value={name} onChange={handleChange} className='model-name-input' />
                  </div>
                  <div>
                    <div className='projecttitle-line'>

                    </div>
                  </div>
                </div>
              </div>
              <div className='set-username-button-field'>
                <div className='create-model-button'>
                  <div onClick={handleSetName} style={{ cursor: 'pointer' }}>
                    <GradationButton text={'登録'} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserNameModal;
