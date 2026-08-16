import React, { useState, useEffect } from "react";
import { Row, Col, Card, Statistic, Table, Tag, Spin } from "antd";
import { 
  TeamOutlined, 
  TrophyOutlined, 
  BarChartOutlined, 
  ThunderboltOutlined 
} from "@ant-design/icons";
import { Line } from "react-chartjs-2";
import { request } from "../api";
import Scoreboard from "./Scoreboard";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    totalPredictions: 0,
    correctPredictions: 0,
    accuracy: 0,
    totalModels: 0,
  });
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    let totalModels = 0; // 提前声明,避免 setStats 引用时处于 TDZ
    try {
      // 获取预测历史
      const predictionsResponse = await request("/api/v1/predictions", {
        headers: {
        },
      });
      
      if (predictionsResponse.ok) {
        const predictionsData = await predictionsResponse.json();
        const predictions = predictionsData.predictions;
        
        // 计算统计信息
        const totalPredictions = predictions.length;
        const correctPredictions = predictions.filter(p => p.is_correct).length;
        const accuracy = totalPredictions > 0 ? (correctPredictions / totalPredictions * 100).toFixed(1) : 0;
        
        setStats({
          totalPredictions,
          correctPredictions,
          accuracy,
          totalModels,
        });

        // 最近预测
        setRecentPredictions(predictions.slice(0, 10));
      }

      // 获取模型数量(真实值,替代硬编码)
      try {
        const modelsResponse = await request("/api/v1/models");
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          totalModels = (modelsData.models || []).length;
        }
      } catch (e) { /* 忽略 */ }

      // 获取模型性能
      const performanceResponse = await request("/api/v1/models/performance");
      if (performanceResponse.ok) {
        const performanceData = await performanceResponse.json();
        setPerformanceData(performanceData.slice(0, 10));
      }
    } catch (error) {
      console.error("获取仪表板数据失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const performanceChartData = {
    labels: performanceData.map((item, index) => `模型 ${index + 1}`),
    datasets: [
      {
        label: "准确率",
        data: performanceData.map(item => item.exact_accuracy * 100),
        borderColor: "#F7C948",
        backgroundColor: "rgba(247, 201, 72, 0.15)",
      },
      {
        label: "Poisson损失",
        data: performanceData.map(item => item.poisson_loss),
        borderColor: "#E5484D",
        backgroundColor: "rgba(229, 72, 77, 0.15)",
      },
    ],
  };

  const performanceChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
        labels: { color: "#93A79B" },
      },
      title: {
        display: true,
        text: "模型性能对比",
        color: "#EFF4ED",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: { color: "rgba(59, 87, 74, 0.35)" },
        ticks: { color: "#93A79B" },
      },
      x: {
        grid: { color: "rgba(59, 87, 74, 0.35)" },
        ticks: { color: "#93A79B" },
      },
    },
  };

  const recentPredictionsColumns = [
    {
      title: "联赛",
      dataIndex: "league_type",
      key: "league_type",
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: "主队",
      dataIndex: "home_team",
      key: "home_team",
      render: (v, r) => r.home_team_zh || v,
    },
    {
      title: "客队",
      dataIndex: "away_team",
      key: "away_team",
      render: (v, r) => r.away_team_zh || v,
    },
    {
      title: "预测比分",
      dataIndex: "predicted_home_goals",
      key: "predicted_home_goals",
      render: (home, record) => `${home}:${record.predicted_away_goals}`,
    },
    {
      title: "实际比分",
      dataIndex: "actual_home_goals",
      key: "actual_home_goals",
      render: (home, record) => home !== null ? `${home}:${record.actual_away_goals}` : "待定",
    },
    {
      title: "结果",
      dataIndex: "is_correct",
      key: "is_correct",
      render: (correct) => correct ? (
        <Tag color="success">正确</Tag>
      ) : (
        <Tag color="error">错误</Tag>
      ),
    },
  ];

  if (loading) {
    return (
      <div className="loading-spinner">
        <Spin size="large" />
      </div>
    );
  }

  const latest = recentPredictions[0];

  return (
    <div>
      <h1 className="page-head">
        比赛日看板
        <small>基于五大联赛统计模型的预测分析席</small>
      </h1>

      {latest ? (
        <div className="matchday-board">
          <div>
            <div className="md-label">今日焦点</div>
            <div className="md-league">
              {latest.league_type === "premier_league" ? "英超" :
               latest.league_type === "la_liga" ? "西甲" :
               latest.league_type === "bundesliga" ? "德甲" :
               latest.league_type === "serie_a" ? "意甲" :
               latest.league_type === "ligue_1" ? "法甲" : latest.league_type}
            </div>
            <Scoreboard
              home={latest.home_team_zh || latest.home_team}
              away={latest.away_team_zh || latest.away_team}
              homeGoals={latest.predicted_home_goals}
              awayGoals={latest.predicted_away_goals}
              size="lg"
              probs={[
                latest.home_win_probability || 0,
                latest.draw_probability || 0,
                latest.away_win_probability || 0,
              ]}
            />
          </div>
          <div className="md-note">
            {latest.is_correct === null
              ? "这场比赛尚未开赛,记分牌展示的是模型预测比分。"
              : latest.is_correct
              ? "预测命中:模型判断与赛果一致。"
              : "预测未命中:赛果偏离了模型判断。"}
          </div>
        </div>
      ) : (
        <div className="matchday-board">
          <div>
            <div className="md-label">比赛日看板</div>
            <div className="md-note">还没有预测记录。去"比赛预测"页生成第一份预测,记分牌会出现在这里。</div>
          </div>
        </div>
      )}
      
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总预测数"
              value={stats.totalPredictions}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="正确预测"
              value={stats.correctPredictions}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="准确率"
              value={stats.accuracy}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: stats.accuracy > 60 ? "#3f8600" : "#cf1322" }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练模型"
              value={stats.totalModels}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="最近预测">
            <Table
              dataSource={recentPredictions}
              columns={recentPredictionsColumns}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="模型性能">
            <div className="chart-container">
              <Line data={performanceChartData} options={performanceChartOptions} />
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
