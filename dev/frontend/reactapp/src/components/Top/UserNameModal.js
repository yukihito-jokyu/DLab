import React, { useContext, useState } from 'react';
import { UserInfoContext } from '../../App';
import { registName } from '../../db/function/users';

function UserNameModal({ setFirstSignIn }) {
  const [name, setName] = useState("");
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  // const { userId, setFirstSignIn } = useContext(UserInfoContext);
  const handleSetName = () => {
    setFirstSignIn(false);
    registName(userId, name)
  }

  return (
    <div className='user-name-modal-wrapper'>
      <div>
        <p>User Name</p>
        <input
          type='text'
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button onClick={handleSetName}>登録</button>
        <button onClick={() => {console.log(userId)}}>id確認</button>
      </div>
    </div>
  );
};

export default UserNameModal;
