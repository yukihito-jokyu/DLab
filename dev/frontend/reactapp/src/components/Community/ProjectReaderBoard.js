import React, { useEffect, useRef, useState } from 'react'
import './Community.css'
import { ReactComponent as SearchIcon } from '../../assets/svg/search_24.svg';
import ReaderBoard from './ReaderBoard';
import { getReaderBoardInfo } from '../../db/function/reader_bord';
import { getFavoriteUser } from '../../db/function/users';
import { useParams } from 'react-router-dom';

function ProjectReaderBoard() {
  const [readerBoard, setReaderBoard] = useState([]);
  const [favorite, setFavorite] = useState(false);
  const [favoriteUser, setFavoriteUser] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredReaderBoard, setFilteredReaderBoard] = useState([]);
  const targetRef = useRef(null);
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const { projectName, task } = useParams();

  useEffect(() => {
    const fetchReaderBoard = async () => {
      const readerBoardData = await getReaderBoardInfo(projectName);
      const favoriteUser = await getFavoriteUser(userId);
      console.log(favoriteUser);
      setReaderBoard(readerBoardData);
      setFavoriteUser(favoriteUser);
      setFilteredReaderBoard(readerBoardData);
    };

    fetchReaderBoard();

  }, [userId, projectName]);

  const handleScroll = () => {
    if (targetRef.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleSearch = () => {
    const filtered = readerBoard.filter(board =>
      board.data().user_name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    setFilteredReaderBoard(filtered);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className='project-reader-board-wrapper'>
      <div className='reader-board-setting'>
        <div className='my-rank-button' onClick={handleScroll} style={{ cursor: 'pointer' }}>
          <p>自分の順位</p>
        </div>
        <div className='favorite-button' onClick={() => setFavorite(!favorite)} style={{ cursor: 'pointer' }}>
          <p>お気に入り</p>
        </div>
        <div className='user-search'>
          <input
            type='text'
            placeholder='ユーザー検索'
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className='search-wrapper' onClick={handleSearch} style={{ cursor: 'pointer' }}>
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
          {task === 'ImageClassification' ? <p>Accuracy</p> : <p>Reward</p>}
        </div>
      </div>
      <div className='reader-board-field'>
        {favorite === true ? (
          filteredReaderBoard && filteredReaderBoard.length > 0 ? (
            filteredReaderBoard.map((board, index) => {
              const userData = board.data();
              return (userData.user_id === userId ? (
                <div key={userData.user_id} ref={targetRef}>
                  <ReaderBoard rank={index + 1} name={userData.user_name} accuracy={userData.accuracy} tileUserId={userData.user_id} />
                </div>
              ) : (
                favoriteUser.includes(userData.user_id) ? (
                  <div key={userData.user_id}>
                    <ReaderBoard rank={index + 1} name={userData.user_name} accuracy={userData.accuracy} tileUserId={userData.user_id} />
                  </div>
                ) : null
              ));
            })
          ) : <p className='none-text'>該当するユーザーが見つかりませんでした。</p>
        ) : (
          filteredReaderBoard && filteredReaderBoard.length > 0 ? (
            filteredReaderBoard.map((board, index) => {
              const userData = board.data();
              return (userData.user_id === userId ? (
                <div key={userData.user_id} ref={targetRef}>
                  <ReaderBoard rank={index + 1} name={userData.user_name} accuracy={userData.accuracy} tileUserId={userData.user_id} />
                </div>
              ) : (
                <div key={userData.user_id}>
                  <ReaderBoard rank={index + 1} name={userData.user_name} accuracy={userData.accuracy} tileUserId={userData.user_id} />
                </div>
              ));
            })
          ) : <p className='none-text'>該当するユーザーが見つかりませんでした。</p>
        )}
      </div>
    </div>
  );
}

export default ProjectReaderBoard;
