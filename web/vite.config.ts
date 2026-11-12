import { defineConfig, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const proxy: Record<string, ProxyOptions> = {
  '/api': {
    target: 'http://127.0.0.1:8080',
    changeOrigin: true,
    proxyTimeout: 10_000,
    configure: server => {
      server.on('proxyReq', request => {
        request.removeHeader('connection')
        request.removeHeader('accept-language')
      })
    },
  },
}

export default defineConfig({
  envDir: false,
  plugins: [react(), tailwindcss()],
  server: { proxy, strictPort: true },
  preview: { proxy, strictPort: true },
})
