import React, { useState } from 'react';
import { apiClient } from '../api/client';
import { Sliders, Play, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

export default function RunSettings({ egoId, onRunComplete }) {
  const [config, setConfig] = useState({
    k: 4,
    seeding: 'weak_clique',
    encoder_type: 'sage',
    hidden_dim: 64,
    lr: 0.01,
    epochs: 300,
    structure_loss: 'bp',
    lambda_seed: 1.0,
    tau: 0.5
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await apiClient.runPipeline(egoId, config);
      setResult(res);
      if (onRunComplete) onRunComplete();
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Pipeline execution failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '840px' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Model Hyperparameters & Execution</h2>
          <p className="section-subtitle">
            Configure GNN encoder options, clique seeding strategy, and loss weights for Ego Network #{egoId}.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.5rem' }}>
          {/* Seeding Strategy */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Seeding Strategy
            </label>
            <select
              value={config.seeding}
              onChange={(e) => setConfig({ ...config, seeding: e.target.value })}
              className="ego-select-dropdown"
              style={{ width: '100%' }}
            >
              <option value="cpm">CPM (k-Clique Percolation)</option>
              <option value="weak_clique">Weak-Clique Seeds (Paper #26 / WOCD)</option>
            </select>
          </div>

          {/* Clique size k */}
          <div style={{ opacity: config.seeding === 'weak_clique' ? 0.45 : 1 }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Clique Size (k): <strong style={{ color: '#38bdf8' }}>{config.k}</strong>
            </label>
            <input
              type="range"
              min="3"
              max="5"
              step="1"
              value={config.k}
              disabled={config.seeding === 'weak_clique'}
              onChange={(e) => setConfig({ ...config, k: parseInt(e.target.value, 10) })}
              style={{ width: '100%', accentColor: 'var(--accent-indigo)' }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{config.seeding === 'weak_clique' ? 'Not used with weak-clique seeds' : 'k=3 (triangles) to k=5 (5-person cliques)'}</span>
          </div>

          {/* GNN Encoder Architecture */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              GNN Encoder Architecture
            </label>
            <select
              value={config.encoder_type}
              onChange={(e) => setConfig({ ...config, encoder_type: e.target.value })}
              className="ego-select-dropdown"
              style={{ width: '100%' }}
            >
              <option value="sage">GraphSAGE (Mean Aggregation)</option>
              <option value="gat">GAT (Graph Attention Network)</option>
            </select>
          </div>

          {/* Structure Loss Objective */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Structure Loss Objective
            </label>
            <select
              value={config.structure_loss}
              onChange={(e) => setConfig({ ...config, structure_loss: e.target.value })}
              className="ego-select-dropdown"
              style={{ width: '100%' }}
            >
              <option value="bp">Bernoulli-Poisson (Default for Overlap)</option>
              <option value="modularity">Soft Modularity Maximization</option>
            </select>
          </div>

          {/* Seed Consistency Weight lambda */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Seed Loss Weight (&lambda;<sub>seed</sub>): <strong style={{ color: '#38bdf8' }}>{config.lambda_seed}</strong>
            </label>
            <input
              type="range"
              min="0.0"
              max="3.0"
              step="0.2"
              value={config.lambda_seed}
              onChange={(e) => setConfig({ ...config, lambda_seed: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-indigo)' }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Higher keeps circles tightly anchored to seeds</span>
          </div>

          {/* Membership Threshold tau */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Membership Threshold (&tau;): <strong style={{ color: '#38bdf8' }}>{config.tau}</strong>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={config.tau}
              onChange={(e) => setConfig({ ...config, tau: parseFloat(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent-indigo)' }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Lower = more overlapping circles per person</span>
          </div>

          {/* Epochs */}
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Training Epochs: <strong style={{ color: '#38bdf8' }}>{config.epochs}</strong>
            </label>
            <input
              type="range"
              min="40"
              max="200"
              step="20"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value, 10) })}
              style={{ width: '100%', accentColor: 'var(--accent-indigo)' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.25rem' }}>
          <button
            type="submit"
            className="btn-primary"
            disabled={loading}
            style={{ opacity: loading ? 0.7 : 1 }}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="spin-icon" /> Training Model & Seeding...
              </>
            ) : (
              <>
                <Play size={16} /> Re-Run CircleSense Pipeline
              </>
            )}
          </button>
        </div>
      </form>

      {/* Success Notification */}
      {result && (
        <div className="glass-panel" style={{ padding: '1.25rem', border: '1px solid var(--accent-emerald)', background: 'rgba(16, 185, 129, 0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#34d399', fontWeight: 600 }}>
            <CheckCircle size={18} />
            <span>Pipeline finished successfully!</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
            Detected <strong>{result.summary?.num_detected_circles} circles</strong> with ONMI = <strong>{result.metrics?.hybrid?.onmi}</strong> and F1 = <strong>{result.metrics?.hybrid?.avg_f1}</strong>.
            You can now view updated circles and visualizations in the other tabs.
          </div>
        </div>
      )}

      {error && (
        <div className="glass-panel" style={{ padding: '1.25rem', border: '1px solid var(--accent-rose)', background: 'rgba(244, 63, 94, 0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fb7185', fontWeight: 600 }}>
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  );
}
