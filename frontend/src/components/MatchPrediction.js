import React, { useState, useEffect } from "react";
import { Form, Select, Input, Button, Card, Row, Col, Statistic, Alert, message, Tag } from "antd";
import { 
  TeamOutlined, 
  TrophyOutlined 
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

const { Option } = Select;

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const MatchPrediction = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [leagues, setLeagues] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchLeagues();
    fetchPredictionHistory();
  }, []);

  const fetchLeagues = async () => {
    try {
      const response = await request("/api/v1/leagues");
      if (response.ok) {
        const data = await response.json();
        setLeagues(data);
      }
    } catch (error) {
      console.error("获取联赛列表失败:", error);
    }
  };

  const fetchPredictionHistory = async () => {
    try {
      const response = await request("/api/v1/predictions", {
        headers: {
        },
      });
      if (response.ok) {
        const data = await response.json();
        setHistory(data.items.slice(0, 5));
      }
    } catch (error) {
      console.error("获取预测历史失败:", error);
    }
  };

  const onFinish = async (values) => {
    setLoading(true);
    try {
      const response = await request("/api/v1/predictions/match", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data.prediction);
        message.success("预测完成");
        fetchPredictionHistory();
      } else {
        const error = await response.json();
        message.error(error.error || "预测失败");
      }
    } catch (error) {
      message.error("网络错误，请重试");
    } finally {
      setLoading(false);
    }
  };

  const predictionChartData = {
    labels: ["主队胜", "平局", "客队胜"],
    datasets: [
      {
        label: "概率",
        data: [
          (prediction?.home_win_probability || 0) * 100,
          (prediction?.draw_probability || 0) * 100,
          (prediction?.away_win_probability || 0) * 100,
        ],
        backgroundColor: [
          "rgba(255, 99, 132, 0.2)",
          "rgba(54, 162, 235, 0.2)",
          "rgba(75, 192, 192, 0.2)",
        ],
        borderColor: [
          "rgb(255, 99, 132)",
          "rgb(54, 162, 235)",
          "rgb(75, 192, 192)",
        ],
        borderWidth: 1,
      },
    ],
  };

  const predictionChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "比赛结果概率",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  };

  return (
    <div className="prediction-form">
      <h1 className="page-head">比赛预测<small>选择联赛与对阵,模型给出比分与胜平负概率</small></h1>
      
      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="新建预测">
            <Form
              form={form}
              layout="vertical"
              onFinish={onFinish}
            >
              <Form.Item
                name="league_type"
                label="联赛类型"
                rules={[{ required: true, message: "请选择联赛类型" }]}
              >
                <Select placeholder="选择联赛类型">
                  {leagues.map(league => (
                    <Option key={league.id} value={league.league_type}>
                      {league.name} ({league.season})
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                name="home_team"
                label="主队"
                rules={[{ required: true, message: "请输入主队名称" }]}
              >
                <Input placeholder="输入主队名称" />
              </Form.Item>

              <Form.Item
                name="away_team"
                label="客队"
                rules={[{ required: true, message: "请输入客队名称" }]}
              >
                <Input placeholder="输入客队名称" />
              </Form.Item>

              <Form.Item
                name="date"
                label="比赛日期"
              >
                <Input type="datetime-local" />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  style={{ width: "100%" }}
                >
                  开始预测
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={8}>
          <Card title="预测历史">
            {history.length > 0 ? (
              <div>
                {history.map((item, index) => (
                  <div key={item.id} style={{ marginBottom: 16 }}>
                    <div>
                      <strong>{item.home_team_zh || item.home_team_zh || m.home_team}</strong> vs <strong>{item.away_team_zh || item.away_team_zh || m.away_team}</strong>
                    </div>
                    <div>
                      预测: {item.predicted_home_goals}:{item.predicted_away_goals}
                    </div>
                    <div>
                      {item.is_correct === null ? (
                        <Tag>待复盘</Tag>
                      ) : item.is_correct ? (
                        <Tag color="success">正确</Tag>
                      ) : (
                        <Tag color="error">错误</Tag>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p>暂无预测历史</p>
            )}
          </Card>
        </Col>
      </Row>

      {prediction && (
        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={14}>
            <Card title="预测结果" style={{ borderLeft: "3px solid #F7C948" }}>
              <Scoreboard
                home={prediction.home_team_zh || prediction.home_team}
                away={prediction.away_team_zh || prediction.away_team}
                homeGoals={prediction.predicted_home_goals}
                awayGoals={prediction.predicted_away_goals}
                size="lg"
                probs={[
                  prediction.home_win_probability || 0,
                  prediction.draw_probability || 0,
                  prediction.away_win_probability || 0,
                ]}
              />
              <div style={{ marginTop: 20 }}>
                <Alert
                  message="预测详情"
                  description={`
                    预测比分: ${prediction.predicted_home_goals}:${prediction.predicted_away_goals}
                    
                    主队胜率: ${((prediction.home_win_probability || 0) * 100).toFixed(1)}%
                    平局概率: ${((prediction.draw_probability || 0) * 100).toFixed(1)}%
                    客队胜率: ${((prediction.away_win_probability || 0) * 100).toFixed(1)}%
                    
                    预测时间: ${new Date().toLocaleString()}
                  `}
                  type="info"
                  showIcon
                />
              </div>
            </Card>
          </Col>

          <Col span={12}>
            <Card title="概率分布">
              <div className="chart-container">
                <Line data={predictionChartData} options={predictionChartOptions} />
              </div>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default MatchPrediction;
