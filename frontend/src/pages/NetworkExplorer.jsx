import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import NetworkGraph from '../components/NetworkGraph';
import NodeDetailPanel from '../components/NodeDetailPanel';
import { Search, Filter, Layers, Info } from 'lucide-react';

export default function NetworkExplorer({ egoId, onSelectForRecommend }) {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeMethod, setActiveMethod] = useState('hybrid');
  const [communities, setCommunities] = useState({});
  const [labels, setLabels] = useState({});
  const [activeCircle, setActiveCircle] = useState('all');
  const [selectedNode, setSelectedNode] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // 1. Fetch graph elements for chosen ego
  useEffect(() => {
    setLoading(true);
    setSelectedNode(null);
    Promise.all([
      apiClient.getGraph(egoId),
      apiClient.getCommunities(egoId, activeMethod)
    ])
      .then(([gData, cData]) => {
        setGraphData(gData);
        setCommunities(cData.circles || {});
        setLabels(cData.labels || {});
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [egoId]);

  // 2. Fetch communities when activeMethod changes
  useEffect(() => {
    apiClient.getCommunities(egoId, activeMethod)
      .then((cData) => {
        setCommunities(cData.circles || {});
        setActiveCircle('all');
      })
      .catch((err) => console.error(err));
  }, [activeMethod, egoId]);

  const circleNames = Object.keys(communities);

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      setSelectedNode(searchQuery.trim());
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Control Header */}
      <div className="glass-panel" style={{ padding: '1rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        {/* Method switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Layers size={15} /> Method:
          </span>
          <div className="nav-tabs" style={{ padding: '0.2rem' }}>
            {[
              { id: 'hybrid', label: 'Hybrid (Ours)' },
              { id: 'cpm', label: 'CPM' },
              { id: 'leiden', label: 'Leiden' },
              { id: 'bigclam', label: 'BigCLAM' },
              { id: 'ground_truth', label: 'Ground Truth' }
            ].map((m) => (
              <button
                key={m.id}
                className={`nav-tab-btn ${activeMethod === m.id ? 'active' : ''}`}
                onClick={() => setActiveMethod(m.id)}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Filter by Circle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Filter size={15} color="var(--text-muted)" />
            <select
              value={activeCircle}
              onChange={(e) => setActiveCircle(e.target.value)}
              className="ego-select-dropdown"
              style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}
            >
              <option value="all">All Circles ({circleNames.length})</option>
              {circleNames.map((c) => (
                <option key={c} value={c}>
                  {labels[c] || c} ({communities[c]?.length || 0} friends)
                </option>
              ))}
            </select>
          </div>

          {/* Search Node */}
          <form onSubmit={handleSearch} style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
            <input
              type="text"
              placeholder="Search Friend ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                background: '#151c2c',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '0.35rem 0.8rem 0.35rem 2rem',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontSize: '0.82rem',
                width: '170px'
              }}
            />
            <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: '0.6rem' }} />
          </form>
        </div>
      </div>

      {/* Network View & Sidebar */}
      <div style={{ display: 'flex', gap: '1rem', minHeight: '620px' }}>
        <div style={{ flex: 1, minHeight: '620px' }}>
          {loading ? (
            <div className="glass-panel" style={{ height: '100%', minHeight: '620px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
              Rendering network topology...
            </div>
          ) : graphData ? (
            <NetworkGraph
              elements={graphData.elements}
              activeCircle={activeCircle}
              selectedNode={selectedNode}
              onNodeSelect={setSelectedNode}
              communityMapping={communities}
            />
          ) : null}
        </div>

        {selectedNode && (
          <NodeDetailPanel
            egoId={egoId}
            nodeId={selectedNode}
            onClose={() => setSelectedNode(null)}
            onSelectForRecommend={onSelectForRecommend}
          />
        )}
      </div>
    </div>
  );
}
