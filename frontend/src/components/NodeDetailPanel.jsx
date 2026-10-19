import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { User, Share2, Award, Sparkles, X } from 'lucide-react';

export default function NodeDetailPanel({ egoId, nodeId, onClose, onSelectForRecommend }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!nodeId) return;
    setLoading(true);
    apiClient.getNodeDetails(egoId, nodeId)
      .then((data) => setDetails(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [egoId, nodeId]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!nodeId) return null;

  return (
    <div className="glass-panel node-panel" style={{
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      borderLeft: '1px solid var(--border-accent)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{
            width: 34,
            height: 34,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <User size={18} color="#fff" />
          </div>
          <div>
            <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Friend #{nodeId}</h4>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Ego Network {egoId}</span>
          </div>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
        >
          <X size={18} />
        </button>
      </div>

      {loading ? (
        <div style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
          Loading profile...
        </div>
      ) : details ? (
        <>
          {/* Status info */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {details.circle_count >= 2 ? (
              <span className="badge badge-bridge">
                <Share2 size={12} /> Bridge Node ({details.circle_count} Circles)
              </span>
            ) : details.circle_count === 1 ? (
              <span className="badge badge-circle">
                <Award size={12} /> 1 Circle Member
              </span>
            ) : (
              <span className="badge" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>
                Periphery Node
              </span>
            )}
          </div>

          {/* Active circles list */}
          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Belongs To Circles
            </label>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.4rem' }}>
              {details.active_circles.length > 0 ? (
                details.active_circles.map((c) => (
                  <span key={c} className="badge badge-circle">
                    {details.labels?.[c] || c}
                  </span>
                ))
              ) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None (Single-link outer friend)</span>
              )}
            </div>
          </div>

          {/* Membership Score Bar Chart */}
          {details.membership_vector && details.membership_vector.length > 0 && (
            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                Strongest circle scores
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginTop: '0.5rem' }}>
                {details.membership_vector
                  .map((score, cIdx) => ({ score, cIdx }))
                  .filter((x) => x.score > 0.01)
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 8)
                  .map(({ score, cIdx }) => (
                  <div key={cIdx} style={{ fontSize: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{details.labels?.[`circle_${cIdx}`] || `Circle ${cIdx}`}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: score >= 0.5 ? '#38bdf8' : 'var(--text-muted)' }}>
                        {score}
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '5px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${Math.min(100, score * 100)}%`,
                        height: '100%',
                        background: score >= 0.5 ? 'linear-gradient(90deg, #6366f1, #38bdf8)' : 'rgba(255,255,255,0.2)'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action button */}
          <button
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', fontSize: '0.82rem', marginTop: '0.5rem' }}
            onClick={() => onSelectForRecommend(nodeId)}
          >
            <Sparkles size={14} /> Friend Recommendations
          </button>
        </>
      ) : null}
    </div>
  );
}
