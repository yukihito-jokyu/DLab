import React, { useContext, useState } from 'react';
import { UserIdContext } from '../../context/context';

function UserNameModal() {
  const [name, setName] = useState("");
  const [userId, setUserId] = useContext(UserIdContext);

  return (
    <div className='user-name-modal-wrapper'>
      <div>
        <p>User Name</p>
        <input
          type='text'
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button>登録</button>
      </div>
    </div>
  );
};

export default UserNameModal;
