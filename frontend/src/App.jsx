import React, { useState, useEffect } from 'react';
import { apiClient } from './api/client';
import Overview from './pages/Overview';
import NetworkExplorer from './pages/NetworkExplorer';
import CirclesView from './pages/CirclesView';
import BridgeFriends from './pages/BridgeFriends';
import FriendSuggestions from './pages/FriendSuggestions';
import CompareMethods from './pages/CompareMethods';
import RunSettings from './pages/RunSettings';
import {
  Share2, Network, Target, Users, BarChart2, Sliders, LayoutDashboard, Cpu
} from 'lucide-react';

export default function App() {
  const [currentTab, setCurrentTab] = useState('overview');
  const [egos, setEgos] = useState([]);
  const [apiError, setApiError] = useState(false);
  const [selectedEgo, setSelectedEgo] = useState(0);
  const [suggestFriendId, setSuggestFriendId] = useState('');

  useEffect(() => {
    apiClient.getEgos()
      .then((data) => {
        setEgos(data);
        if (data.length > 0 && !data.some((e) => e.ego_id === selectedEgo)) {
          setSelectedEgo(data[0].ego_id);
        }
      })
      .catch((err) => {
        console.error('Failed to load egos:', err);
        setApiError(true);
      });
  }, []);

  const handleSelectForRecommend = (nodeId) => {
    setSuggestFriendId(nodeId);
    setCurrentTab('recommend');
  };

  const handleInspectNode = (nodeId) => {
    setCurrentTab('explorer');
  };

  return (
    <div>
      {/* Top Header Bar */}
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div className="logo-badge" onClick={() => setCurrentTab('overview')} style={{ cursor: 'pointer' }}>
            <Network size={22} color="#38bdf8" />
            <span>CircleSense</span>
          </div>

          {/* Ego Network Selector */}
          <div className="ego-selector-bar">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Ego Network:
            </span>
            <select
              value={selectedEgo}
              onChange={(e) => setSelectedEgo(parseInt(e.target.value, 10))}
              className="ego-select-dropdown"
            >
              {egos.map((e) => (
                <option key={e.ego_id} value={e.ego_id}>
                  Ego #{e.ego_id} ({e.nodes} friends, {e.edges} edges)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${currentTab === 'overview' ? 'active' : ''}`}
            onClick={() => setCurrentTab('overview')}
          >
            <LayoutDashboard size={15} /> Overview
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'explorer' ? 'active' : ''}`}
            onClick={() => setCurrentTab('explorer')}
          >
            <Network size={15} /> Graph Explorer
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'circles' ? 'active' : ''}`}
            onClick={() => setCurrentTab('circles')}
          >
            <Target size={15} /> Circles
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'bridges' ? 'active' : ''}`}
            onClick={() => setCurrentTab('bridges')}
          >
            <Share2 size={15} /> Bridge Friends
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'recommend' ? 'active' : ''}`}
            onClick={() => setCurrentTab('recommend')}
          >
            <Users size={15} /> Suggestions
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'compare' ? 'active' : ''}`}
            onClick={() => setCurrentTab('compare')}
          >
            <BarChart2 size={15} /> Benchmark
          </button>
          <button
            className={`nav-tab-btn ${currentTab === 'settings' ? 'active' : ''}`}
            onClick={() => setCurrentTab('settings')}
          >
            <Sliders size={15} /> Settings
          </button>
        </nav>
      </header>

      {/* Main Container */}
      <main className="main-container">
        {apiError && egos.length === 0 && (
          <div className="glass-panel" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem', borderColor: '#f87171' }}>
            <strong style={{ color: '#f87171' }}>Cannot reach the backend API.</strong>{' '}
            Start it with <code>python -m uvicorn app.main:app --app-dir backend --port 8000</code> and reload this page.
          </div>
        )}
        {currentTab === 'overview' && (
          <Overview
            egos={egos}
            selectedEgo={selectedEgo}
            onSelectEgo={setSelectedEgo}
            onNavigate={setCurrentTab}
          />
        )}
        {currentTab === 'explorer' && (
          <NetworkExplorer
            egoId={selectedEgo}
            onSelectForRecommend={handleSelectForRecommend}
          />
        )}
        {currentTab === 'circles' && (
          <CirclesView
            egoId={selectedEgo}
          />
        )}
        {currentTab === 'bridges' && (
          <BridgeFriends
            egoId={selectedEgo}
            onInspectNode={handleInspectNode}
            onSelectForRecommend={handleSelectForRecommend}
          />
        )}
        {currentTab === 'recommend' && (
          <FriendSuggestions
            egoId={selectedEgo}
            selectedFriendId={suggestFriendId}
            onSelectFriendId={setSuggestFriendId}
          />
        )}
        {currentTab === 'compare' && (
          <CompareMethods
            egoId={selectedEgo}
          />
        )}
        {currentTab === 'settings' && (
          <RunSettings
            egoId={selectedEgo}
            onRunComplete={() => {
              // Refresh ego data
              apiClient.getEgos().then((data) => setEgos(data));
            }}
          />
        )}
      </main>
    </div>
  );
}
