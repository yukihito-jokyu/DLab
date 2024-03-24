import React, { useState, useEffect } from 'react';

function DraggableElement({ parentRef }) {
  const [isDragging, setIsDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e) => {
    setIsDragging(true);
    const parentRect = parentRef.current.getBoundingClientRect();
    const offsetX = e.clientX - parentRect.left - position.x;
    const offsetY = e.clientY - parentRect.top - position.y;
    setOffset({ x: offsetX, y: offsetY });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const parentRect = parentRef.current.getBoundingClientRect();
      const newX = e.clientX - parentRect.left - offset.x;
      const newY = e.clientY - parentRect.top - offset.y;

      const elemRect = {
        top: newY,
        bottom: newY + 100,
        left: newX,
        right: newX + 100
      };

      // ドラッグしている要素と他の要素が重ならないかをチェック
      const children = parentRef.current.children;
      let isOverlapping = false;
      for (let i = 0; i < children.length; i++) {
        if (children[i] !== e.target) {
          const childRect = children[i].getBoundingClientRect();
          if (
            elemRect.right > childRect.left &&
            elemRect.left < childRect.right &&
            elemRect.bottom > childRect.top &&
            elemRect.top < childRect.bottom
          ) {
            isOverlapping = true;
            break;
          }
        }
      }

      // 重なりがなければ位置を更新
      if (!isOverlapping) {
        setPosition({ x: newX, y: newY });
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        cursor: isDragging ? 'grabbing' : 'grab',
        backgroundColor: 'lightblue',
        width: 100,
        height: 100,
        border: '1px solid black',
        zIndex: isDragging ? 9999 : 'auto'
      }}
      onMouseDown={handleMouseDown}
    >
      Drag me!
    </div>
  );
}

export default DraggableElement;
