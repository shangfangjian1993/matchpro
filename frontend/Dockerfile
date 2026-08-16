# 多阶段构建:真实 React 源码 → 静态产物(Vite)
FROM node:18-alpine AS build
WORKDIR /app
# 可选:开发模式直连后端地址;生产走 server.js 同源代理(不传即同源 /api)
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
COPY package.json package-lock.json ./
RUN npm ci --registry=https://registry.npmmirror.com --no-audit --no-fund
COPY . .
RUN npm run build

# 运行阶段:静态托管 + API 同源代理 + SPA 路由回退
FROM node:18-alpine
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/build ./build
COPY server.js ./
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
