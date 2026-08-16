const express = require("express");
const path = require("path");
const http = require("http");

const app = express();

// API 同源反向代理:前端 /api/* → 后端 api 容器(compose 网络内服务名)。
// 解决"前端写死 localhost:8000 导致跨设备访问网络错误"的问题——
// 无论从本机、NAS IP 还是局域网设备访问前端,API 请求都走同源 /api。
const API_TARGET = process.env.API_TARGET || "api";   // compose 服务名;本地开发可设 localhost
const API_PORT = process.env.API_PORT || 8000;

app.use("/api", (req, res) => {
  // 转发客户端真实 IP:API 端限流按 X-Forwarded-For 识别(链式代理时追加)。
  const upstreamXff = req.headers["x-forwarded-for"];
  const xff = upstreamXff
    ? `${upstreamXff}, ${req.socket.remoteAddress}`
    : req.socket.remoteAddress;
  const proxyReq = http.request(
    {
      hostname: API_TARGET,
      port: API_PORT,
      path: req.originalUrl,
      method: req.method,
      headers: { ...req.headers, host: `${API_TARGET}:${API_PORT}`, "x-forwarded-for": xff },
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    }
  );
  proxyReq.on("error", (err) => {
    console.error("API 代理错误:", err.message);
    res.status(502).json({ error: "后端服务不可用" });
  });
  req.pipe(proxyReq);
});

// 安全响应头(防 MIME 嗅探/点击劫持/信息泄露;CSP 允许 antd 内联样式)
app.use((req, res, next) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "DENY");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'"
  );
  next();
});

// 静态资源:React 构建产物
app.use(express.static(path.join(__dirname, "build")));

// SPA 回退:浏览器深链(如 /dashboard)刷新时返回 index.html
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "build", "index.html"));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
