import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Sparkles, User, Users, CheckCircle, ArrowRight, CornerDownRight } from 'lucide-react';

export default function FriendSuggestions({ egoId, selectedFriendId, onSelectFriendId }) {
  const [friendId, setFriendId] = useState(selectedFriendId || '');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // If selectedFriendId prop changes, sync local state and fetch
  useEffect(() => {
    if (selectedFriendId) {
      setFriendId(selectedFriendId);
      fetchRecommendations(selectedFriendId);
    }
  }, [selectedFriendId, egoId]);

  const fetchRecommendations = (id) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    apiClient.getRecommendations(egoId, id, 10)
      .then((data) => {
        setRecommendations(data.recommendations || []);
      })
      .catch((err) => {
        console.error(err);
        setError(`Unable to generate suggestions for Friend ${id}.`);
        setRecommendations([]);
      })
      .finally(() => setLoading(false));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (friendId.trim()) {
      onSelectFriendId(friendId.trim());
      fetchRecommendations(friendId.trim());
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Smart Friend Suggestions</h2>
          <p className="section-subtitle">
            Explainable friend recommendations derived from learned overlapping community affinity (F_u &middot; F_v).
          </p>
        </div>
      </div>

      {/* Target friend selector */}
      <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, maxWidth: '480px' }}>
          <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
            Generate Suggestions For:
          </label>
          <input
            type="text"
            placeholder="Enter Friend ID (e.g. 1)"
            value={friendId}
            onChange={(e) => setFriendId(e.target.value)}
            style={{
              background: '#151c2c',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '0.45rem 0.9rem',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.88rem',
              flex: 1
            }}
          />
          <button type="submit" className="btn-primary" style={{ padding: '0.45rem 1rem', fontSize: '0.85rem' }}>
            <Sparkles size={15} /> Find Matches
          </button>
        </form>

        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Tip: You can also click "Suggest" from the Bridge Friends table or Network Explorer.
        </span>
      </div>

      {/* Results Container */}
      {loading ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          Calculating shared-circle dot product affinity...
        </div>
      ) : error ? (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: '#fb7185' }}>
          {error}
        </div>
      ) : recommendations.length === 0 ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          Enter a friend ID above and click "Find Matches" to view personalized friend recommendations.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            Showing top {recommendations.length} recommended connections for <strong>Friend #{friendId}</strong>:
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1rem' }}>
            {recommendations.map((rec, idx) => (
              <div
                key={rec.target_node}
                className="glass-panel"
                style={{
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '0.9rem'
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'linear-gradient(135deg, #06b6d4, #6366f1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <User size={18} color="#fff" />
                      </div>
                      <div>
                        <h4 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Friend #{rec.target_node}</h4>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Match Rank #{idx + 1}</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Affinity Score</span>
                      <div style={{ fontSize: '1.2rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                        {rec.affinity_score}
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Users size={14} color="var(--text-muted)" />
                    <span>Shares {rec.common_neighbors_count} mutual friends in network</span>
                  </div>

                  {/* Shared circles breakdown */}
                  <div>
                    <label style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                      Shared Circles Justification
                    </label>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.35rem' }}>
                      {rec.shared_circles?.map((c) => (
                        <div
                          key={c.circle_id}
                          style={{
                            background: '#151c2c',
                            padding: '0.35rem 0.6rem',
                            borderRadius: '6px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: '0.75rem'
                          }}
                        >
                          <span style={{ color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <CornerDownRight size={11} /> {c.label || c.circle_name}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                            contrib: <strong style={{ color: '#34d399' }}>+{c.affinity_contribution}</strong>
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.7rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                    <CheckCircle size={12} color="#10b981" /> Explainable link
                  </span>
                  <button
                    className="btn-secondary"
                    onClick={() => {
                      setFriendId(rec.target_node);
                      onSelectFriendId(rec.target_node);
                      fetchRecommendations(rec.target_node);
                    }}
                    style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
                  >
                    Pivot to Friend #{rec.target_node} <ArrowRight size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
