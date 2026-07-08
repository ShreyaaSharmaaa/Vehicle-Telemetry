import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxying /api -> the FastAPI backend means the frontend code never needs
// to hardcode http://localhost:8000, and this same proxy pattern is what
// you'd configure in nginx/a reverse proxy in a real deployment -- so the
// dev setup mirrors the eventual production setup instead of diverging
// from it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
