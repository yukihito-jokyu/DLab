import './ImageClassificationProjectList.css'
import { useNavigate } from 'react-router-dom'
import CIFAR10Image1 from '../../assets/images/project_image/CIFAR10/CIFAR10_image1.png';
import CIFAR10Image2 from '../../assets/images/project_image/CIFAR10/CIFAR10_image2.png';
import CIFAR10Image3 from '../../assets/images/project_image/CIFAR10/CIFAR10_image3.png';
import CIFAR100Image1 from '../../assets/images/project_image/CIFAR100/CIFAR100_image1.png';
import CIFAR100Image2 from '../../assets/images/project_image/CIFAR100/CIFAR100_image2.png';
import CIFAR100Image3 from '../../assets/images/project_image/CIFAR100/CIFAR100_image3.png';
import FashionMNISTImage1 from '../../assets/images/project_image/FashionMNIST/FashionMNIST_image1.png';
import FashionMNISTImage2 from '../../assets/images/project_image/FashionMNIST/FashionMNIST_image2.png';
import FashionMNISTImage3 from '../../assets/images/project_image/FashionMNIST/FashionMNIST_image3.png';
import MNISTImage1 from '../../assets/images/project_image/MNIST/MNIST_image1.png';
import MNISTImage2 from '../../assets/images/project_image/MNIST/MNIST_image2.png';
import MNISTImage3 from '../../assets/images/project_image/MNIST/MNIST_image3.png';

function ImageProjectIcon({ projectName }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const navigate = useNavigate();
  const handleNav = (projectName) => {
    let task;
    if (projectName === "CartPole" || projectName === "FlappyBird") {
      task = "ReinforcementLearning";
    } else {
      task = "ImageClassification";
    }
    // projectNameをセッションに保存
    sessionStorage.setItem('projectId', JSON.stringify(projectName));

    // navigateで遷移先を指定
    navigate(`/ModelManegementEvaluation/${userId}/${task}/${projectName}`);
  };

  const style1 = {
    width: '150px',
    height: '150px'
  };
  const style2 = {
    width: '70px',
    height: '70px'
  };

  return (
    <div className='ImageProjectIcon-wrapper' onClick={() => handleNav(projectName)} style={{ cursor: 'pointer' }}>
      <div className='titlefeild-wrapper'>
        <div className='project-title'>
          <div className='projecttitle-wrapper'>
            <p>{projectName}</p>
          </div>
        </div>
        <div className='projecttitle-line'></div>
      </div>
      <div className='projectimages-wrapper'>
        <div className='big-image'>
          <div style={style1}>
            {projectName === 'CIFAR10' ? (
              <img src={CIFAR10Image1} alt='cifar10' className='image1' />
            ) : projectName === 'CIFAR100' ? (
              <img src={CIFAR100Image1} alt='cifar100' className='image1' />
            ) : projectName === 'FashionMNIST' ? (
              <img src={FashionMNISTImage1} alt='fashionmnist' className='image1' />
            ) : projectName === 'MNIST' ? (
              <img src={MNISTImage1} alt='mnist' className='image1' />
            ) : (
              <></>
            )}
          </div>
        </div>
        <div className='small-image'>
          <div className='is-first'>
            <div style={style2}>
              {projectName === 'CIFAR10' ? (
                <img src={CIFAR10Image2} alt='cifar10' className='image2' />
              ) : projectName === 'CIFAR100' ? (
                <img src={CIFAR100Image2} alt='cifar100' className='image2' />
              ) : projectName === 'FashionMNIST' ? (
                <img src={FashionMNISTImage2} alt='fashionmnist' className='image2' />
              ) : projectName === 'MNIST' ? (
                <img src={MNISTImage2} alt='mnist' className='image2' />
              ) : (
                <></>
              )}
            </div>
          </div>
          <div>
            <div style={style2}>
              {projectName === 'CIFAR10' ? (
                <img src={CIFAR10Image3} alt='cifar10' className='image2' />
              ) : projectName === 'CIFAR100' ? (
                <img src={CIFAR100Image3} alt='cifar100' className='image2' />
              ) : projectName === 'FashionMNIST' ? (
                <img src={FashionMNISTImage3} alt='fashionmnist' className='image2' />
              ) : projectName === 'MNIST' ? (
                <img src={MNISTImage3} alt='mnist' className='image2' />
              ) : (
                <></>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ImageProjectIcon;
