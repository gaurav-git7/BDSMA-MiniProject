import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Share2, User, ExternalLink, Sparkles, Award } from 'lucide-react';

export default function BridgeFriends({ egoId, onInspectNode, onSelectForRecommend }) {
  const [bridges, setBridges] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiClient.getBridges(egoId, 50)
      .then((data) => setBridges(data.bridges || []))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [egoId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Bridge Friends Detection</h2>
          <p className="section-subtitle">
            Friends who simultaneously belong to multiple social circles, serving as social connectors across distinct groups.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="badge badge-bridge" style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}>
            <Share2 size={13} /> {bridges.length} Bridges Discovered
          </span>
        </div>
      </div>

      {loading ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          Extracting multi-community bridge friends...
        </div>
      ) : bridges.length === 0 ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          No bridge friends detected for this ego network under current threshold.
        </div>
      ) : (
        <div className="glass-panel" style={{ overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Rank</th>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Friend ID</th>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Circle Count</th>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Overlapping Circles</th>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Affinity Strength</th>
                  <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600, textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {bridges.map((b, idx) => (
                  <tr
                    key={b.node_id}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      transition: 'background 0.15s ease'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '0.9rem 1rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      #{idx + 1}
                    </td>
                    <td style={{ padding: '0.9rem 1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(244, 63, 94, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <User size={14} color="#fb7185" />
                        </div>
                        <strong style={{ fontFamily: 'var(--font-mono)' }}>Friend {b.node_id}</strong>
                      </div>
                    </td>
                    <td style={{ padding: '0.9rem 1rem' }}>
                      <span className="badge badge-bridge">
                        {b.circle_count} Circles
                      </span>
                    </td>
                    <td style={{ padding: '0.9rem 1rem' }}>
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                        {b.circles.map((c) => (
                          <span
                            key={c.circle_id}
                            style={{
                              background: 'rgba(99, 102, 241, 0.12)',
                              color: '#a5b4fc',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                              fontSize: '0.75rem',
                              fontFamily: 'var(--font-mono)'
                            }}
                          >
                            {c.circle_name} ({c.score})
                          </span>
                        ))}
                      </div>
                    </td>
                    <td style={{ padding: '0.9rem 1rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', fontWeight: 600 }}>
                      {b.total_strength}
                    </td>
                    <td style={{ padding: '0.9rem 1rem', textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '0.4rem' }}>
                        <button
                          className="btn-secondary"
                          style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                          onClick={() => onInspectNode(b.node_id)}
                          title="View on network graph"
                        >
                          <ExternalLink size={12} /> Graph
                        </button>
                        <button
                          className="btn-secondary"
                          style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                          onClick={() => onSelectForRecommend(b.node_id)}
                          title="Generate friend suggestions"
                        >
                          <Sparkles size={12} color="#818cf8" /> Suggest
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
