import React, { useContext } from 'react'
import './ImageClassificationProjectList.css'
import ImageProjectImages from './ImageProjectImages'
import { ReactComponent as EditSVG } from '../../assets/svg/edit.svg' 
import { UserInfoContext } from '../../App'
import { useNavigate } from 'react-router-dom'

function ImageProjectIcon({ projectName }) {
  const { setProjectId } = useContext(UserInfoContext);
  const navigate = useNavigate();
  const handleNav = (projectName) => {
    setProjectId(projectName);
    sessionStorage.setItem('projectId', JSON.stringify(projectName));
    navigate('/ModelManegementEvaluation');
  }
  const style1 = {
    width: '150px',
    height: '150px'
  };
  const style2 = {
    width: '70px',
    height: '70px'
  }
  return (
    <div className='ImageProjectIcon-wrapper' onClick={() => {handleNav(projectName)}}>
      <div className='titlefeild-wrapper'>
        <div className='project-title'>
          <div className='projecttitle-wrapper'>
            <p>{projectName}</p>
          </div>
          <div className='rename-icon'>
            <EditSVG className='edit-svg' />
          </div>
        </div>
        <div className='projecttitle-line'></div>
      </div>
      <div className='projectimages-wrapper'>
        <div className='big-image'>
          <div style={style1}>
            写真
          </div>
        </div>
        <div className='small-image'>
          <div className='is-first'>
            <div style={style2}>
              写真
            </div>
          </div>
          <div>
            <div style={style2}>
              写真
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageProjectIcon
