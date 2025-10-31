import axios from 'axios';

export const nodeApi = axios.create({
  baseURL: (import.meta as any).env.VITE_NODE_API_URL || (typeof __NODE_API_URL__ !== 'undefined' ? __NODE_API_URL__ : 'http://localhost:4000'),
  timeout: 30000,
});

export const fastApi = axios.create({
  baseURL: (import.meta as any).env.VITE_FASTAPI_URL || (typeof __FASTAPI_URL__ !== 'undefined' ? __FASTAPI_URL__ : 'http://localhost:8000'),
  timeout: 300000,
});



