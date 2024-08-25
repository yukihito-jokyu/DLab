import React, { useEffect, useState } from 'react'
import './ImageClassificationProjectList.css'
import ImageProjectIcon from './ImageProjectIcon'
import ImageProjectAdd from './ImageProjectAdd'
import BorderGradationBox from '../../uiParts/component/BorderGradationBox'
import { getJoinProject } from '../../db/function/users'


function ImageProjectField() {
  const [participationProjects, setParticipationProjects] = useState(null);
  const style1 = {
    margin: '0px 40px 40px 0',
    width: '276px',
    height: '241px'
  };
  // 参加プロジェクトをfirebaseから取得する
  useEffect(() => {
    const fatchProjects = async () => {
      const userId = JSON.parse(sessionStorage.getItem('userId'));
      const projects = await getJoinProject(userId);
      setParticipationProjects(projects);
    };
    fatchProjects();
  }, []);
  return (
    <div className='imageprojectfield-wrapper'>
      {participationProjects ? (
          participationProjects.sort().map((value, index) => (
            <div key={index}>
              <BorderGradationBox style1={style1}>
                <ImageProjectIcon projectName={value.project_name} />
              </BorderGradationBox>
            </div>
          ))
        ) : (<></>)}
      <ImageProjectAdd />
    </div>
  )
}

export default ImageProjectField
