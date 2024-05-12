import React from 'react';
import '../css/CreateButton.css'
import GradationButton from './GradationButton';

function CreateButton() {
  const text = '作成';
  return (
    <div>
      <GradationButton text={text} />
    </div>
  );
};

export default CreateButton;
