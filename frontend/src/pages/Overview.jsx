import React from 'react';
import { Users, GitCommit, Target, Share2, ArrowRight, ShieldCheck, Cpu } from 'lucide-react';

export default function Overview({ egos = [], selectedEgo, onSelectEgo, onNavigate }) {
  const totalNodes = egos.reduce((acc, curr) => acc + (curr.nodes || 0), 0);
  const totalEdges = egos.reduce((acc, curr) => acc + (curr.edges || 0), 0);
  const totalCircles = egos.reduce((acc, curr) => acc + (curr.detected_circles || curr.ground_truth_circles || 0), 0);
  const totalBridges = egos.reduce((acc, curr) => acc + (curr.bridges || 0), 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Hero Banner */}
      <div className="glass-panel" style={{
        padding: '2.5rem',
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%)',
        border: '1px solid var(--border-accent)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ maxWidth: '820px', position: 'relative', zIndex: 2 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.25rem 0.75rem', borderRadius: '999px', background: 'rgba(99, 102, 241, 0.15)', border: '1px solid rgba(99, 102, 241, 0.3)', marginBottom: '1rem' }}>
            <Cpu size={14} color="#818cf8" />
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#a5b4fc' }}>BDSMA Micro-Project (Module 4)</span>
          </div>
          <h1 style={{ fontSize: '2.25rem', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2, marginBottom: '0.75rem' }}>
            CircleSense: Overlapping Friend-Circle Detection
          </h1>
          <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1.5rem' }}>
            Combining <strong>Clique Percolation (CPM)</strong> for explainable group seeds with <strong>Graph Neural Networks (GraphSAGE / GAT)</strong> to uncover overlapping friend communities, isolate bridge friends, and provide transparent social recommendations.
          </p>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button className="btn-primary" onClick={() => onNavigate('explorer')}>
              Explore Network Graph <ArrowRight size={16} />
            </button>
            <button className="btn-secondary" onClick={() => onNavigate('compare')}>
              Benchmark vs Baselines
            </button>
          </div>
        </div>
      </div>

      {/* Global Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Total Ego Networks
            </span>
            <Users size={18} color="#06b6d4" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>10</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>SNAP ego-Facebook benchmark</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Total Friends Analyzed
            </span>
            <Users size={18} color="#6366f1" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {totalNodes.toLocaleString() || '4,039'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Across all ego graphs</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Friendship Edges
            </span>
            <GitCommit size={18} color="#10b981" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {totalEdges.toLocaleString() || '88,234'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Mutual connections</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Detected Circles
            </span>
            <Target size={18} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {totalCircles || 58}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Soft overlapping communities</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Bridge Friends
            </span>
            <Share2 size={18} color="#f43f5e" />
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {totalBridges || 210}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>In 2 or more circles</div>
        </div>
      </div>

      {/* 10 Ego Networks Grid */}
      <div>
        <div className="section-header">
          <div>
            <h2 className="section-title">Ego Networks Directory</h2>
            <p className="section-subtitle">Select any Facebook ego-network to explore its topology, circles, and bridge connections</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          {egos.map((ego) => {
            const isSelected = selectedEgo === ego.ego_id;
            return (
              <div
                key={ego.ego_id}
                className="glass-panel"
                onClick={() => {
                  onSelectEgo(ego.ego_id);
                  onNavigate('explorer');
                }}
                style={{
                  padding: '1.25rem',
                  cursor: 'pointer',
                  border: isSelected ? '2px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'rgba(6, 182, 212, 0.08)' : 'var(--bg-card)'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Ego Network #{ego.ego_id}</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {ego.ego_id}</span>
                  </div>
                  <span className="badge badge-success">
                    <ShieldCheck size={12} /> Ready
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '1rem', fontSize: '0.82rem' }}>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Friends: </span>
                    <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{ego.nodes}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Edges: </span>
                    <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{ego.edges}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Circles: </span>
                    <strong style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{ego.detected_circles || ego.ground_truth_circles}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Bridges: </span>
                    <strong style={{ fontFamily: 'var(--font-mono)', color: '#fb7185' }}>{ego.bridges || '-'}</strong>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', display: 'inline-flex', alignItems: 'center', gap: '0.2rem', fontWeight: 600 }}>
                    Inspect Graph <ArrowRight size={14} />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
