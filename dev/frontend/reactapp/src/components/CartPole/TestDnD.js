import React, { useRef } from 'react';
import DraggableElement from './DraggableElement';

function TestDnD() {
  const parentRef = useRef(null);
  return (
    <div
      ref={parentRef}
      style={{
        position: 'relative',
        width: '100vw',
        height: '50vh',
        border: '1px solid black',
        overflow: 'hidden' // 親要素からはみ出るのを防ぐためにoverflowをhiddenに設定します
      }}
    >
      <DraggableElement parentRef={parentRef} />
      <DraggableElement parentRef={parentRef} />
    </div>
  );
}

export default TestDnD;
