import React, { useContext, useEffect, useState } from 'react'
import './ImageClassificationProjectList.css'
import ImageProjectIcon from './ImageProjectIcon'
import ImageProjectAdd from './ImageProjectAdd'
import BorderGradationBox from '../../uiParts/component/BorderGradationBox'
import { UserInfoContext } from '../../App'
import { getJoinProject, getProject } from '../../db/firebaseFunction'


function ImageProjectField() {
  // const { userId } = useContext(UserInfoContext);
  const [participationProjects, setParticipationProjects] = useState(null);
  const style1 = {
    margin: '0px 40px 40px 0',
    width: '276px',
    height: '241px'
  };
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
          participationProjects.sort().map((name, index) => (
            <div key={index}>
              <BorderGradationBox style1={style1}>
                <ImageProjectIcon projectName={name} />
              </BorderGradationBox>
            </div>
          ))
        ) : (<></>)}
      <ImageProjectAdd />
    </div>
  )
}

export default ImageProjectField
