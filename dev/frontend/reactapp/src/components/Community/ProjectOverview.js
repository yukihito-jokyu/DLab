import React from 'react';
import './Community.css';

function ProjectOverview() {
  return (
    <div className='project-overview-wrapper'>
      <div className='overview-wrapper'>
        <p className='overview-title'>1. Summary</p>
        <p className='overview-info'>The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.</p>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>2. Data format</p>
        <p className='overview-info'>データの形式は、以下の通りです。</p>
        <div></div>
      </div>
      <div className='overview-wrapper'>
        <p className='overview-title'>3. Source</p>
        <p className='overview-info'>このプロジェクトは、”The CIFAR-10 dataset”(https://www.cs.toronto.edu/~kriz/cifar.html)で配布されているデータを用いています。</p>
      </div>
    </div>
  );
};

export default ProjectOverview
