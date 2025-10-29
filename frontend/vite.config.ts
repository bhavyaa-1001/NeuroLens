import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    define: {
      __APP_VERSION__: JSON.stringify('0.1.0'),
      __NODE_API_URL__: JSON.stringify(env.VITE_NODE_API_URL || 'http://localhost:4000'),
      __FASTAPI_URL__: JSON.stringify(env.VITE_FASTAPI_URL || 'http://localhost:8000')
    }
  };
});


