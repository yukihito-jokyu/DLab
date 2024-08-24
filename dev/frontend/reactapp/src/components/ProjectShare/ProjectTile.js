import React, { useContext } from 'react';
import './ProjectShare.css';
import CIFAR10Tile from '../../assets/images/CIFAR10Tile.png';
import CIFAR100Tile from '../../assets/images/CIFAR100Tile.png'
import MNISTTile from '../../assets/images/MNISTTile.png';
import FashionMNISTTile from '../../assets/images/FashionMNISTTile.png';
import CartPoleTile from '../../assets/images/CartPoleTile.png';
import FlappyBirdTile from '../../assets/images/FlappyBirdTile.png';
import { UserInfoContext } from '../../App';
import { useNavigate } from 'react-router-dom';

function ProjectTile({ title, info, style1, style2 }) {
  const { setProjectId } = useContext(UserInfoContext);
  const navigate = useNavigate();

  let imageSrc, altText;
  switch (title) {
    case 'CIFAR10':
      imageSrc = CIFAR10Tile;
      altText = 'CIFAR10 Tile';
      break;
    case 'CIFAR100':
      imageSrc = CIFAR100Tile;
      altText = 'CIFAR100 Tile';
      break;
    case 'MNIST':
      imageSrc = MNISTTile;
      altText = 'MNIST Tile';
      break;
    case 'FashionMNIST':
      imageSrc = FashionMNISTTile;
      altText = 'FashionMNIST Tile';
      break;
    case 'CartPole':
      imageSrc = CartPoleTile;
      altText = 'CartPole Tile';
      break;
    case 'FlappyBird':
      imageSrc = FlappyBirdTile;
      altText = 'FlappyBird Tile';
      break;
    default:
      imageSrc = CIFAR10Tile;
      altText = 'Default Tile';
  }

  const handleNav = () => {
    setProjectId(title);
    sessionStorage.setItem('projectId', JSON.stringify(title));
    navigate(`/community/${title}`);
  }

  return (
    <div className='project-tile-wrapper' onClick={handleNav} style={{ cursor: 'pointer' }}>
      <div className='tile-wrapper'>
        <div className='tile-left' style={style1}>
          <p className='project-tile-title'>{title}</p>
          <p dangerouslySetInnerHTML={{ __html: info }} className='project-tile-info'></p>
        </div>
        <div className='tile-right'>
          <img src={imageSrc} alt={altText} className='image' />
          <div className='image-cover' style={style2}></div>
        </div>
      </div>
    </div>
  );
};

export default ProjectTile;
