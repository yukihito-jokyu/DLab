// Reactアプリのコード (frontend/src/App.jsなど)
import React, { useEffect, useState } from 'react';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      console.log('test2');
      try {
        const response = await fetch('http://127.0.0.1:5000/');  // Flaskアプリのエンドポイントに合わせてURLを設定
        console.log(response);
        const result = await response.json();
        console.log(result);
        setData(result);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      <h1>Data from Flask:</h1>
      <ul>
        {data.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;