import React, { useState, useEffect } from "react";
import { Routes, Route, Navigate, useNavigate, useLocation } from "react-router-dom";
import { Layout, Menu, message, Button, ConfigProvider, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import {
  TeamOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  TrophyOutlined,
  LogoutOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { request } from "./api";
import Login from "./components/Login";
import Register from "./components/Register";
import Dashboard from "./components/Dashboard";
import MatchPrediction from "./components/MatchPrediction";
import TournamentPrediction from "./components/TournamentPrediction";
import ModelTraining from "./components/ModelTraining";
import ModelPerformance from "./components/ModelPerformance";
import PredictionHistory from "./components/PredictionHistory";
import Settings from "./components/Settings";

const { Header, Sider, Content } = Layout;

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // 会话状态经 httpOnly cookie 承载:启动时调 /me 恢复(无 token 则未登录)
    request("/api/v1/auth/me")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.user) {
          setIsAuthenticated(true);
          setUser(data.user);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const handleLogin = (userData) => {
    setIsAuthenticated(true);
    setUser(userData);
    message.success("登录成功");
  };

  const handleRegister = (userData) => {
    setUser(userData);
    message.success("注册成功");
  };

  const themeConfig = {
    algorithm: antdTheme.darkAlgorithm,
    token: {
      colorPrimary: "#F7C948",
      colorBgBase: "#0A1511",
      colorBgContainer: "#101F18",
      colorBgElevated: "#101F18",
      colorBorder: "#1B2E24",
      colorBorderSecondary: "#1B2E24",
      colorText: "#EFF4ED",
      colorTextSecondary: "#93A79B",
      colorTextTertiary: "#3D574A",
      colorSuccess: "#3DD68C",
      colorError: "#E5484D",
      borderRadius: 2,
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
    },
    components: {
      Layout: { siderBg: "#101F18", headerBg: "#101F18" },
      Menu: { darkItemBg: "transparent", darkItemSelectedBg: "#F7C948", darkItemSelectedColor: "#0A1511" },
    },
  };

  const handleLogout = () => {
    // 通知后端清除 httpOnly cookie,再重置本地状态
    request("/api/v1/auth/logout", { method: "POST" });
    setIsAuthenticated(false);
    setUser(null);
    message.success("退出成功");
  };

  if (loading) {
    return <div className="loading-spinner">加载中...</div>;
  }

  if (!isAuthenticated) {
    return (
      <ConfigProvider locale={zhCN} theme={themeConfig}>
        <Routes>
          <Route path="/login" element={<Login onLogin={handleLogin} />} />
          <Route path="/register" element={<Register onRegister={handleRegister} />} />
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </ConfigProvider>
    );
  }

  return (
    <ConfigProvider locale={zhCN} theme={themeConfig}>
    <Layout style={{ minHeight: "100vh" }}>
      <Sider collapsible>
        <div className="brand">
          <div className="brand-mark">⚽</div>
          <span className="brand-name">
            预测<em>席</em>
          </span>
        </div>
        <Menu
          theme="dark"
          selectedKeys={[location.pathname]}
          mode="inline"
          onClick={({ key }) => navigate(key)}
          items={[
            {
              key: "/dashboard",
              icon: <BarChartOutlined />,
              label: "比赛日看板",
            },
            {
              key: "/match-prediction",
              icon: <ThunderboltOutlined />,
              label: "比赛预测",
            },
            {
              key: "/tournament-prediction",
              icon: <TrophyOutlined />,
              label: "赛事预测",
            },
            {
              key: "/model-training",
              icon: <SettingOutlined />,
              label: "模型训练",
            },
            {
              key: "/model-performance",
              icon: <BarChartOutlined />,
              label: "模型性能",
            },
            {
              key: "/prediction-history",
              icon: <TeamOutlined />,
              label: "预测历史",
            },
            {
              key: "/settings",
              icon: <SettingOutlined />,
              label: "设置",
            },
          ]}
        />
      </Sider>
      <Layout>
        <Header>
          <span className="header-kicker">Matchday Analytics</span>
          <div className="header-user">
            <span>
              欢迎, <span className="user-name">{user?.username}</span>
            </span>
            <Button
              type="primary"
              icon={<LogoutOutlined />}
              onClick={handleLogout}
            >
              退出
            </Button>
          </div>
        </Header>
        <Content>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/match-prediction" element={<MatchPrediction />} />
            <Route path="/tournament-prediction" element={<TournamentPrediction />} />
            <Route path="/model-training" element={<ModelTraining />} />
            <Route path="/model-performance" element={<ModelPerformance />} />
            <Route path="/prediction-history" element={<PredictionHistory />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
    </ConfigProvider>
  );
}

export default App;
