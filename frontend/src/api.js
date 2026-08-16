// 统一 API 访问封装:默认同源 /api(生产由 server.js 反向代理到后端),
// 开发模式可用 VITE_API_URL 覆盖(如 http://localhost:8000)。
// 认证经 httpOnly cookie(JS 不可读);非 GET 请求携带 CSRF 双提交令牌。

export const API_BASE = import.meta.env.VITE_API_URL || "";

// 从 cookie 读取 CSRF 令牌(flask-jwt-extended 双提交防护)
function getCsrfToken() {
  const m = document.cookie.match(/(?:^|; )csrf_access_token=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : "";
}

export async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  // 非 GET 请求带 CSRF 头(cookie 模式认证必需)
  const method = (options.method || "GET").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers["X-CSRF-TOKEN"] = getCsrfToken();
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });

  // 会话过期:跳转登录
  if (response.status === 401 && !path.startsWith("/api/v1/auth/login")) {
    if (window.location.pathname !== "/login") {
      window.location.href = "/login";
    }
  }
  return response;
}
