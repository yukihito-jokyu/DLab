import React, { createContext, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css'
import ImageClassificationProjectList from './components/ImageClassificationProjectList/ImageClassificationProjectList';
import ModelManegementEvaluation from './components/ModelManegementEvaluation/ModelManegementEvaluation';
import ModelCreateTrain from './components/ModelCreateTrain/ModelCreateTrain';
import RLProjectList from './components/RLProjectList/RLProjectList';
import Top from './components/Top/Top';
import ProjectShare from './components/ProjectShare/ProjectShare';
import Community from './components/Community/Community';
import Profile from './components/Profile/Profile';


export const UserInfoContext = createContext()

function App() {
  const [userId, setUserId] = useState('');
  const [projectId, setProjectId] = useState("");
  const [firstSignIn, setFirstSignIn] = useState(false);

  return (
    <UserInfoContext.Provider value={{ userId, setUserId, firstSignIn, setFirstSignIn, projectId, setProjectId }}>
      <Router>
        <Routes>
          <Route path='/top' element={<Top />} />
          <Route path="/ImageClassificationProjectList/:userId" element={<ImageClassificationProjectList />} />
          <Route path="/ModelManegementEvaluation/:userId/:task/:projectName" element={<ModelManegementEvaluation />} />
          <Route path="/ModelCreateTrain/:task/:projectName/:modelId" element={<ModelCreateTrain />} />
          <Route path="/RLProjectList" element={<RLProjectList />} />
          <Route path='/projectshare' element={<ProjectShare />} />
          <Route path='/community/:task/:projectName' element={<Community />} />
          <Route path='/profile/:profileUserId' element={<Profile />} />
          <Route path="*" element={<Navigate to="/top" replace />} />
        </Routes>
      </Router>
    </UserInfoContext.Provider>
  );
}

export default App;