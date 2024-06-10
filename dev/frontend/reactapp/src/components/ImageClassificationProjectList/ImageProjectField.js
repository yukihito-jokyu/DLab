import React, { useContext, useEffect, useState } from 'react'
import './ImageClassificationProjectList.css'
import ImageProjectIcon from './ImageProjectIcon'
import ImageProjectAdd from './ImageProjectAdd'
import BorderGradationBox from '../../uiParts/component/BorderGradationBox'
import { UserInfoContext } from '../../App'
import { getProject } from '../../db/firebaseFunction'


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
      const projects = await getProject(userId);
      setParticipationProjects(projects);
    };

    fatchProjects();

  }, []);
  return (
    <div className='imageprojectfield-wrapper'>
      {participationProjects ? (
          Object.keys(participationProjects).sort().map((key) => {
            if (participationProjects[key] === true && key !== 'user_id') {
              return (
                <div key={key}>
                  <BorderGradationBox style1={style1}>
                    <ImageProjectIcon projectName={key} />
                  </BorderGradationBox>
                  {/* {key}: {participationProjects[key].toString()} */}
                </div>
              );
            } else {
              return null
            }
          })
        ) : (<></>)}
      <ImageProjectAdd />
    </div>
  )
}

export default ImageProjectField
