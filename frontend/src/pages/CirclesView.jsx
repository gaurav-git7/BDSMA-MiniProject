import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Target, Info, Sparkles, Layers, Activity, ChevronRight, X } from 'lucide-react';

export default function CirclesView({ egoId }) {
  const [circlesData, setCirclesData] = useState({});
  const [labels, setLabels] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCircle, setSelectedCircle] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [expLoading, setExpLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    setSelectedCircle(null);
    setExplanation(null);
    apiClient.getCommunities(egoId, 'hybrid')
      .then((data) => {
        setCirclesData(data.circles || {});
        setLabels(data.labels || {});
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [egoId]);

  const handleExplain = (circleName) => {
    setSelectedCircle(circleName);
    setExpLoading(true);
    apiClient.getCircleExplanation(egoId, circleName)
      .then((data) => setExplanation(data))
      .catch((err) => console.error(err))
      .finally(() => setExpLoading(false));
  };

  const circleEntries = Object.entries(circlesData);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Detected Circles & Explanations</h2>
          <p className="section-subtitle">
            Inspect discovered communities, their internal cohesiveness, and the clique seeds they expanded from.
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedCircle ? '1fr 380px' : '1fr', gap: '1.5rem' }}>
        {/* Circles Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem', alignContent: 'start' }}>
          {loading ? (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              Loading friend circles...
            </div>
          ) : circleEntries.length === 0 ? (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              No circles detected yet for this network.
            </div>
          ) : (
            circleEntries.map(([cName, members], idx) => {
              const isSelected = selectedCircle === cName;
              return (
                <div
                  key={cName}
                  className="glass-panel"
                  style={{
                    padding: '1.25rem',
                    border: isSelected ? '2px solid var(--accent-indigo)' : '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '1rem'
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#6366f1' }}></div>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, textTransform: 'capitalize' }}>
                          {labels[cName] || cName.replace('_', ' ')}
                        </h3>
                      </div>
                      <span className="badge badge-circle">{members.length} Friends</span>
                    </div>

                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      Represents a cohesive social sub-community grown outward from dense core friendships.
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                    <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap', maxWidth: '160px' }}>
                      {members.slice(0, 4).map((m) => (
                        <span key={m} style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.05)', padding: '0.1rem 0.4rem', borderRadius: '4px', fontFamily: 'var(--font-mono)' }}>
                          #{m}
                        </span>
                      ))}
                      {members.length > 4 && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>+{members.length - 4}</span>
                      )}
                    </div>

                    <button
                      className="btn-secondary"
                      onClick={() => handleExplain(cName)}
                      style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                    >
                      <Sparkles size={13} color="#a5b4fc" /> Explain
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Explanation Panel */}
        {selectedCircle && (
          <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem', height: 'fit-content' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={18} color="#818cf8" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Circle Explanation{selectedCircle && labels[selectedCircle] ? `: ${labels[selectedCircle]}` : ''}</h3>
              </div>
              <button
                onClick={() => setSelectedCircle(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            {expLoading ? (
              <div style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                Analyzing seed origins and graph metrics...
              </div>
            ) : explanation ? (
              <>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {explanation.summary}
                </p>

                {/* Cohesion metrics */}
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                    Graph Cohesion
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.4rem' }}>
                    <div style={{ background: '#151c2c', padding: '0.6rem', borderRadius: '8px' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Internal Density</span>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#34d399' }}>
                        {explanation.quality?.density ?? '-'}
                      </div>
                    </div>
                    <div style={{ background: '#151c2c', padding: '0.6rem', borderRadius: '8px' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Conductance (Cut)</span>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#fb7185' }}>
                        {explanation.quality?.conductance ?? '-'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Seed Cliques Origin */}
                {explanation.seed_origin && (
                  <div>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                      Clique Seed Origin
                    </label>
                    <div style={{ background: '#151c2c', padding: '0.75rem', borderRadius: '8px', marginTop: '0.4rem', fontSize: '0.82rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Type:</span>
                        <strong style={{ color: '#a5b4fc' }}>{explanation.seed_origin.seed_type}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Initial Seed Size:</span>
                        <strong>{explanation.seed_origin.seed_size} members</strong>
                      </div>
                      {explanation.seed_origin.constituent_cliques?.length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Sample Core Cliques:</span>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', marginTop: '0.2rem' }}>
                            {explanation.seed_origin.constituent_cliques.slice(0, 3).map((clique, i) => (
                              <code key={i} style={{ fontSize: '0.72rem', background: 'rgba(0,0,0,0.3)', padding: '0.15rem 0.4rem', borderRadius: '4px' }}>
                                [{clique.join(', ')}]
                              </code>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Top distinctive features */}
                {explanation.top_features && explanation.top_features.length > 0 && (
                  <div>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                      Distinctive Profile Features
                    </label>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginTop: '0.4rem' }}>
                      {explanation.top_features.map((feat) => (
                        <div key={feat.feature_id} style={{ background: '#151c2c', padding: '0.5rem 0.75rem', borderRadius: '6px', fontSize: '0.78rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-primary)' }}>{feat.name}</span>
                            <span style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>+{Math.round(feat.distinctiveness * 100)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
