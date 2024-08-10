import React, { useContext } from 'react'
import './ImageClassificationProjectList.css'
import ImageProjectImages from './ImageProjectImages'
import { ReactComponent as EditSVG } from '../../assets/svg/edit.svg'
import { UserInfoContext } from '../../App'
import { useNavigate } from 'react-router-dom'
import CIFAR10Image1 from '../../assets/images/project_image/CIFAR10/CIFAR10_image1.png';
import CIFAR10Image2 from '../../assets/images/project_image/CIFAR10/CIFAR10_image2.png';
import CIFAR10Image3 from '../../assets/images/project_image/CIFAR10/CIFAR10_image3.png';

function ImageProjectIcon({ projectName }) {
  // const { setProjectId } = useContext(UserInfoContext);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const navigate = useNavigate();
  const handleNav = (projectName) => {
    // setProjectId(projectName);
    sessionStorage.setItem('projectId', JSON.stringify(projectName));
    navigate(`/ModelManegementEvaluation/${userId}/${projectName}`);
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
    <div className='ImageProjectIcon-wrapper' onClick={() => { handleNav(projectName) }} style={{ cursor: 'pointer' }}>
      <div className='titlefeild-wrapper'>
        <div className='project-title'>
          <div className='projecttitle-wrapper'>
            <p>{projectName}</p>
          </div>
          {/* <div className='rename-icon'>
            <EditSVG className='edit-svg' />
          </div> */}
        </div>
        <div className='projecttitle-line'></div>
      </div>
      <div className='projectimages-wrapper'>
        <div className='big-image'>
          <div style={style1}>
            <img src={CIFAR10Image1} alt='cifar10' className='image1' />
          </div>
        </div>
        <div className='small-image'>
          <div className='is-first'>
            <div style={style2}>
              <img src={CIFAR10Image2} alt='cifar10' className='image2' />
            </div>
          </div>
          <div>
            <div style={style2}>
              <img src={CIFAR10Image3} alt='cifar10' className='image2' />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageProjectIcon
