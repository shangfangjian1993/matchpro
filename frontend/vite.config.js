// Vite 配置:构建产物输出到 build/(与 server.js 静态托管/SPA 回退兼容);
// 开发模式 /api 代理到后端(默认 localhost:8000,可用 VITE_DEV_API_TARGET 覆盖)。
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // CRA 迁移:源码为 .js 但含 JSX,让 esbuild 按 JSX 解析 .js 文件
  esbuild: {
    loader: "jsx",
    include: /src\/.*\.jsx?$/,
    exclude: [],
  },
  build: {
    outDir: "build",
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
