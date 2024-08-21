import React, { useEffect, useState } from 'react';
import './Community.css';
import GradationButton from '../../uiParts/component/GradationButton';
import CIFAR10Image from '../../assets/images/CIFAR10Image.png';
import CIFAR100Image from '../../assets/images/CIFAR100Image.png';
import MNISTImage from '../../assets/images/MNISTImage.png';
import FashionMNISTImage from '../../assets/images/FashionMNISTImage.png';
import CartPoleImage from '../../assets/images/CartPoleImage.png';
import FlappyBirdImage from '../../assets/images/FlappyBirdImage.png';

import { useNavigate } from 'react-router-dom';
import { getJoinProject } from '../../db/function/users';

function ProjectActivate({ projectName, shortExp, changeJoinModal }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [joined, setJoined] = useState(false);

  useEffect(() => {
    const fetchJoinProject = async () => {
      const joinProject = await getJoinProject(userId);
      if (joinProject) {
        if (joinProject.some(project => project.project_name === projectName)) {
          setJoined(true);
        } else {
          setJoined(false);
        }
      }
      if (projectName === 'CartPole' || projectName === 'FlappyBird') {
        setJoined(true);
      }
    };
    fetchJoinProject();
  }, [userId, projectName]);

  // projectNameに応じて画像とaltタグを設定
  let imageSrc, altText;
  switch (projectName) {
    case 'CIFAR10':
      imageSrc = CIFAR10Image;
      altText = 'CIFAR10 Image';
      break;
    case 'CIFAR100':
      imageSrc = CIFAR100Image;
      altText = 'CIFAR100 Image';
      break;
    case 'MNIST':
      imageSrc = MNISTImage;
      altText = 'MNIST Image';
      break;
    case 'FashionMNIST':
      imageSrc = FashionMNISTImage;
      altText = 'FashionMNIST Image';
      break;
    case 'CartPole':
      imageSrc = CartPoleImage;
      altText = 'CartPole Image';
      break;
    case 'FlappyBird':
      imageSrc = FlappyBirdImage;
      altText = 'FlappyBird Image';
      break;
    default:
      imageSrc = CIFAR10Image;
      altText = 'Default Project Image';
  }

  const style1 = {
    width: '200px',
    background: 'linear-gradient(95.34deg, #B6F862 3.35%, #00957A 100%), linear-gradient(94.22deg, #D997FF 0.86%, #50BCFF 105.96%)'
  };
  const navigate = useNavigate();

  const handleActivate = () => {
    sessionStorage.setItem('projectId', JSON.stringify(projectName));
    let task;
    if (projectName === "CartPole" || projectName === "FlappyBird") {
      task = "ReinforcementLearning";
    } else {
      task = "ImageClassification";
    }
    navigate(`/ModelManegementEvaluation/${userId}/${task}/${projectName}`);
  }

  return (
    <div className='project-activate-wrapper'>
      <div className='activate-left'>
        {projectName ? (
          <div className='activate-info-field'>
            <p className='activate-project-title'>{projectName}</p>
            <p dangerouslySetInnerHTML={{ __html: shortExp }} className='activate-project-info'></p>
          </div>
        ) : (<></>)}
        {joined ? (
          <div className='activate-button-wrapper' onClick={handleActivate} style={{ cursor: 'pointer' }}>
            <GradationButton text={'Activate'} style1={style1} />
          </div>
        ) : (
          <div className='activate-button-wrapper' onClick={changeJoinModal} style={{ cursor: 'pointer' }}>
            <GradationButton text={'join'} style1={style1} />
          </div>
        )}
      </div>
      <div className='activate-right'>
        <img src={imageSrc} alt={altText} className='activate-project-image' />
      </div>
    </div>
  );
};

export default ProjectActivate;
