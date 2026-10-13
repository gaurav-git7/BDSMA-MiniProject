import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid
} from 'recharts';
import { Trophy, Clock, CheckCircle2, Zap } from 'lucide-react';

export default function CompareMethods({ egoId }) {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiClient.getMetrics(egoId)
      .then((data) => setMetricsData(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [egoId]);

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Gathering comparative benchmarks...
      </div>
    );
  }

  const rawMetrics = metricsData?.metrics || {};
  const recEval = metricsData?.recommendation_eval || {};

  const methods = [
    { key: 'hybrid', label: 'Hybrid (Ours)' },
    { key: 'cpm', label: 'CPM' },
    { key: 'leiden', label: 'Leiden' },
    { key: 'bigclam', label: 'BigCLAM' },
    { key: 'gnn_no_seeds', label: 'GNN (No Seeds)' }
  ];

  // Chart data for accuracy metrics
  const accuracyChartData = methods.map((m) => {
    const data = rawMetrics[m.key] || {};
    return {
      name: m.label,
      ONMI: data.onmi || 0,
      'Avg F1': data.avg_f1 || 0,
      'Omega Index': data.omega || 0
    };
  });

  // Chart data for runtimes
  const runtimeChartData = methods.map((m) => {
    const data = rawMetrics[m.key] || {};
    return {
      name: m.label,
      'Runtime (s)': data.runtime || 0
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div className="section-header">
        <div>
          <h2 className="section-title">Comparative Performance Benchmark</h2>
          <p className="section-subtitle">
            Evaluating CircleSense against established baselines (CPM, Leiden, BigCLAM, and seedless GNN) on Ego Network #{egoId}.
          </p>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '1rem 1.25rem', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Trophy size={18} color="#f59e0b" />
          <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Ground-Truth Agreement & Bridge Metrics</h3>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.02)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Algorithm</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>ONMI</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Avg F1</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Omega Index</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Bridge Precision</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Bridge Recall</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Bridge F1</th>
                <th style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)', textAlign: 'right' }}>Runtime</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((m) => {
                const item = rawMetrics[m.key] || {};
                const isHybrid = m.key === 'hybrid';
                return (
                  <tr
                    key={m.key}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      background: isHybrid ? 'rgba(99, 102, 241, 0.06)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <strong style={{ color: isHybrid ? '#38bdf8' : 'var(--text-primary)' }}>
                        {m.label}
                      </strong>
                      {isHybrid && <span className="badge badge-success" style={{ marginLeft: '0.5rem' }}>Proposed</span>}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: isHybrid ? 700 : 400 }}>
                      {item.onmi ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: isHybrid ? 700 : 400 }}>
                      {item.avg_f1 ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: isHybrid ? 700 : 400 }}>
                      {item.omega ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)' }}>
                      {item.bridge_precision ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)' }}>
                      {item.bridge_recall ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', color: '#fb7185' }}>
                      {item.bridge_f1 ?? '-'}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', textAlign: 'right', color: 'var(--text-secondary)' }}>
                      {item.runtime ? `${item.runtime}s` : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem' }}>
        {/* Accuracy Comparison Bar Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Accuracy Metrics (ONMI, Avg F1, Omega Index)</h4>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyChartData} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: '#101522', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Bar dataKey="ONMI" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Avg F1" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Omega Index" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Runtime Comparison Bar Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Execution Runtime (Seconds)</h4>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={runtimeChartData} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: '#101522', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="Runtime (s)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recommendation Link Recovery Hit-Rate */}
      {recEval.hybrid_hit_rate !== undefined && (
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
              <Zap size={18} color="#06b6d4" />
              <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Recommendation Quality: Hidden Edge Recovery</h4>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              10% of edges were randomly hidden. Hit-rate@10 measures the success rate of recovering true friendships.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <div style={{ background: '#151c2c', padding: '0.8rem 1.25rem', borderRadius: '10px', textAlign: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>CircleSense Hit-Rate</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                {Math.round(recEval.hybrid_hit_rate * 100)}%
              </div>
            </div>
            <div style={{ background: '#151c2c', padding: '0.8rem 1.25rem', borderRadius: '10px', textAlign: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Common Neighbors Baseline</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                {Math.round((recEval.cn_hit_rate || 0) * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
