import React, { useEffect, useRef, useState } from 'react'
import './Community.css'
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg';
import ReaderBoard from './ReaderBoard';
import { getReaderBoardInfo } from '../../db/function/reader_bord';
import { getFavoriteUser } from '../../db/function/users';

function ProjectReaderBoard() {
  const [readerBoard, setReaderBoard] = useState([]);
  const [favorite, setFavorite] = useState(false);
  const [favoriteUser, setFavoriteUser] = useState([]);
  const targetRef = useRef(null);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  useEffect(() => {
    const fetchReaderBoard = async () => {
      const projectId = JSON.parse(sessionStorage.getItem('projectId'));
      const readerBoardData = await getReaderBoardInfo(projectId);
      const favoriteUser = await getFavoriteUser(userId);
      setReaderBoard(readerBoardData);
      setFavoriteUser(favoriteUser);
    };

    fetchReaderBoard();

  }, [userId]);

  const handleScroll = () => {
    // targetRef.currentが存在する場合にスクロール
    if (targetRef.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };
  return (
    <div className='project-reader-board-wrapper'>
      <div className='reader-board-setting'>
        <div className='my-rank-button' onClick={handleScroll}>
          <p>自分の順位</p>
        </div>
        <div className='favorite-button' onClick={() => setFavorite(!favorite)}>
          <p>お気に入り</p>
        </div>
        <div className='user-search'>
          <input
            type='text'
            placeholder='ユーザー検索'
          />
          <div className='search-wrapper'>
            <SearchIcon className='search-icon' />
          </div>
        </div>
      </div>
      <div className='reader-board-header'>
        <div className='reader-board-rank'>
          <p>Rank</p>
        </div>
        <div className='reader-board-name'>
          <p>User Name</p>
        </div>
        <div className='reader-board-eval'>
          <p>Accuracy</p>
        </div>
      </div>
      <div className='reader-board-field'>
        {favorite === true ? (
          readerBoard ? (
            readerBoard.map((board) => {
              const userData = board.data();
              return (userData.user_id === userId ? (
                <div key={userData.user_id} ref={targetRef}>
                  <ReaderBoard rank={userData.rank} name={userData.user_name} accuracy={userData.accuracy} />
                </div>
              ) : (
                favoriteUser.includes(userData.user_name) ? (
                  <div key={userData.user_id}>
                    <ReaderBoard rank={userData.rank} name={userData.user_name} accuracy={userData.accuracy} />
                  </div>
                ) : null
              ));
            })
          ) : null
        ) : (
          readerBoard ? (
            readerBoard.map((board) => {
              const userData = board.data();
              return (userData.user_id === userId ? (
                <div key={userData.user_id} ref={targetRef}>
                  <ReaderBoard rank={userData.rank} name={userData.user_name} accuracy={userData.accuracy} />
                </div>
              ) : (
                <div key={userData.user_id}>
                  <ReaderBoard rank={userData.rank} name={userData.user_name} accuracy={userData.accuracy} />
                </div>
              ));
            })
          ) : null
        )}
        
      </div>
    </div>
  )
}

export default ProjectReaderBoard;
