import React, { useEffect, useState } from 'react'
import ParticipationTile from './ParticipationTile';
import { useNavigate, useParams } from 'react-router-dom';
import { addFavorite, getFavoriteUser, getJoinProject, getRegistrationDate, getUserName, handlSignOut, removeFavorite } from '../../db/function/users';
import './Profile.css';
import { ReactComponent as CheckSVG } from '../../assets/svg/check_24.svg'; 
import GradationButton from '../../uiParts/component/GradationButton';

function ProfileField() {
  const { profileUserId } = useParams();
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [userName , setUserName] = useState();
  const [joinProject, setJoinProject] = useState();
  const [otherProfile, setOtherProfile] = useState();
  const [favoriteUser, setFavaoriteUser] = useState();
  const [registrationDate, setRegistrationDate] = useState();
  const navigate = useNavigate();
  useEffect(() => {
    const fetchFirebase = async () => {
      const name = await getUserName(profileUserId);
      const projectList = await getJoinProject(profileUserId);
      const favoriteUserList = await getFavoriteUser(userId);
      const date = await getRegistrationDate(profileUserId);
      setUserName(name);
      setJoinProject(projectList);
      console.log(favoriteUserList.some(id => id === profileUserId))
      setFavaoriteUser(favoriteUserList.some(id => id === profileUserId));
      setOtherProfile(profileUserId !== userId);
      setRegistrationDate(date)
    }
    fetchFirebase();
  }, [profileUserId, userId]);
  const formatTimestamp = (timestamp) => {
    if (timestamp && timestamp.seconds) {
      const date = new Date(timestamp.seconds * 1000);
      return date.toLocaleDateString(); // 日付と時刻をローカル形式で表示
    }
    return '';
  };
  const style1 = {
    border: favoriteUser ? '5px solid #F0B927' : 'none'
  }
  const style2 = {
    backgroundColor: favoriteUser ? '#D9D9D9' : '#F0B927'
  }
  const handleFavorite = async () => {
    if (favoriteUser) {
      await removeFavorite(userId, profileUserId);
      setFavaoriteUser(!favoriteUser);
    } else {
      await addFavorite(userId, profileUserId);
      setFavaoriteUser(!favoriteUser);
    }
  }

  const handleNav = () => {
    handlSignOut();
    navigate('/top');
  }
  return (
    <div className='profile-field-wrapper'>
      <div className='profile-title'>
        <p>Profile</p>
      </div>
      <div className='profile'>
        <div className='profile-left'>
          <div className='picture-field' style={style1}></div>
        </div>
        <div className='profile-right'>
          <div className='profile-info'>
            <p className='user-name'>{userName}</p>
            <p className='registration-date'>登録日：{formatTimestamp(registrationDate)}</p>
          </div>
        </div>
      </div>
      <div className='profile-project-list-wrapper'>
        <div className='project-list-header'>
          <div className='project-header'>
            <p>Projects</p>
          </div>
          <div className='rank-header'>
            <p>Rank</p>
          </div>
        </div>
        <div className='project-list-field'>
          {joinProject ? (
            joinProject.map((value, index) => (
              <div key={index}>
                <ParticipationTile projectName={value.project_name} rank={value.rank} />
              </div>
            ))
          ) : (
            <></>
          )}
        </div>
      </div>
      {otherProfile ? (
        <div className='favorite-button-wrapper' style={style2} onClick={handleFavorite}>
          {favoriteUser ? (
            <CheckSVG className='check-svg' />
          ) : (
            <p>★</p>
          )}
        </div>
      ) : (
        <></>
      )}
      {!otherProfile ? (
        <div className='logout-wrapper'>
          <div onClick={handleNav}>
            <GradationButton text={'LOGOUT'} />
          </div>
        </div>
      ) : (
        <></>
      )}
    </div>
  )
}

export default ProfileField;
