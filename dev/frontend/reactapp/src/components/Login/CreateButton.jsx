import React from 'react';
import './CreateButton.css'
import GradationButton from '../../uiParts/component/GradationButton';

function CreateButton() {
  const text = '作成';
  return (
    <div>
      <GradationButton text={text} />
    </div>
  );
};

export default CreateButton;
