import axios from 'axios';

const API_BASE = '/api';

export const apiClient = {
  getEgos: async () => {
    const res = await axios.get(`${API_BASE}/egos`);
    return res.data;
  },

  getGraph: async (egoId) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/graph`);
    return res.data;
  },

  getCommunities: async (egoId, method = 'hybrid') => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/communities`, {
      params: { method }
    });
    return res.data;
  },

  getNodeDetails: async (egoId, nodeId) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/nodes/${nodeId}`);
    return res.data;
  },

  getBridges: async (egoId, top = 30) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/bridges`, {
      params: { top }
    });
    return res.data;
  },

  getRecommendations: async (egoId, nodeId, top = 10) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/recommend/${nodeId}`, {
      params: { top }
    });
    return res.data;
  },

  getCircleExplanation: async (egoId, circleName) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/circles/${circleName}/explain`);
    return res.data;
  },

  getMetrics: async (egoId) => {
    const res = await axios.get(`${API_BASE}/egos/${egoId}/metrics`);
    return res.data;
  },

  runPipeline: async (egoId, config) => {
    const res = await axios.post(`${API_BASE}/egos/${egoId}/run`, config);
    return res.data;
  }
};
