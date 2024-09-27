import React from 'react'
import './ProjectShare.css';

function ProjectShareHeader({ handleClickImage, handleClickReinforcement, image, reinforcement, joinProject, handleJoin }) {
  const style1 = {
    height: image ? '54px' : '50px',
    backgroundColor: image ? 'white' : '#D9D9D9',
    borderTop: image ? 'solid 4px #03B0CD' : 'none',
    borderLeft: image ? 'solid 4px #03B0CD' : 'none',
    borderRight: image ? 'solid 4px #03B0CD' : 'none',
    color: image ? '#03B0CD' : '#868686',
    cursor: 'pointer'
  }
  const style2 = {
    height: reinforcement ? '54px' : '50px',
    backgroundColor: reinforcement ? 'white' : '#D9D9D9',
    borderTop: reinforcement ? 'solid 4px #FF5D99' : 'none',
    borderLeft: reinforcement ? 'solid 4px #FF5D99' : 'none',
    borderRight: reinforcement ? 'solid 4px #FF5D99' : 'none',
    color: reinforcement ? '#FF5D99' : '#868686',
    cursor: 'pointer'
  }
  const style3 = {
    backgroundColor: image ? '#03B0CD' : '#FF5D99'
  }
  const style4 = {
    backgroundColor: !joinProject ? '#D9D9D9' : '#c6f6ff',
    cursor: 'pointer'
  }
  const style5 = {
    backgroundColor: image ? '#03B0CD' : '#FF5D99',
    left: joinProject ? 'calc(100% - 30px)' : '0'
  }
  return (
    <div className='project-share-header'>
      <div className='project-share-header-wrapper'>
        <div className='project-share-header-item' onClick={handleClickImage} style={style1}>
          <p>画像分類</p>
        </div>
        <div className='project-share-header-item' onClick={handleClickReinforcement} style={style2}>
          <p>強化学習</p>
        </div>
      </div>
      <div className='project-share-header-bottom' style={style3}></div>
      <div className='display-button-wrapper'>
        {image && (<div className='wrapper'>
          <p>参加済みのプロジェクトを表示</p>
          <div className='display-button-cover' onClick={handleJoin} style={style4}>
            <div className='display-button' style={style5}></div>
          </div>
        </div>)}
      </div>
    </div>
  )
}

export default ProjectShareHeader;
