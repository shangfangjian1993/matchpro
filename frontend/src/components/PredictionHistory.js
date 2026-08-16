import React, { useState, useEffect } from "react";
import { Card, Table, Tag, Button, Modal, Statistic, Row, Col, DatePicker, Select } from "antd";
import { 
  TeamOutlined, 
  TrophyOutlined, 
  BarChartOutlined, 
  EyeOutlined,
  ReloadOutlined 
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
const { RangePicker } = DatePicker;

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PredictionHistory = () => {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [selectedLeague, setSelectedLeague] = useState(null);
  const [dateRange, setDateRange] = useState(null);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);

  useEffect(() => {
    fetchPredictions();
  }, []);

  useEffect(() => {
    filterPredictions();
  }, [predictions, selectedLeague, dateRange]);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await request("/api/v1/predictions", {
        headers: {
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setPredictions(data.items);
      }
    } catch (error) {
      console.error("获取预测历史失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const filterPredictions = () => {
    let filtered = [...predictions];

    if (selectedLeague) {
      filtered = filtered.filter(item => item.league_type === selectedLeague);
    }

    if (dateRange) {
      const startDate = new Date(dateRange[0]);
      const endDate = new Date(dateRange[1]);
      filtered = filtered.filter(item => {
        const itemDate = new Date(item.prediction_timestamp);
        return itemDate >= startDate && itemDate <= endDate;
      });
    }

    setFilteredPredictions(filtered);
  };

  const showPredictionDetail = (prediction) => {
    setSelectedPrediction(prediction);
    setModalVisible(true);
  };

  const getAccuracyStats = () => {
    // 只统计已复盘记录(is_correct 非 null 的比赛),未赛预测不进入准确率
    const reviewed = filteredPredictions.filter(p => p.is_correct !== null);
    if (reviewed.length === 0) return null;
    
    const correctCount = reviewed.filter(p => p.is_correct).length;
    const totalCount = reviewed.length;
    const accuracy = (correctCount / totalCount * 100).toFixed(1);
    
    const withinOneCount = reviewed.filter(p => 
      Math.abs(p.predicted_home_goals - p.actual_home_goals) <= 1 &&
      Math.abs(p.predicted_away_goals - p.actual_away_goals) <= 1
    ).length;
    
    const withinTwoCount = reviewed.filter(p => 
      Math.abs(p.predicted_home_goals - p.actual_home_goals) <= 2 &&
      Math.abs(p.predicted_away_goals - p.actual_away_goals) <= 2
    ).length;

    return {
      total: totalCount,
      correct: correctCount,
      accuracy: accuracy,
      withinOne: withinOneCount,
      withinTwo: withinTwoCount,
    };
  };

  const getLeagueStats = () => {
    const leagueStats = {};
    
    filteredPredictions.forEach(prediction => {
      if (prediction.is_correct === null) return; // 未复盘不计入联赛统计
      if (!leagueStats[prediction.league_type]) {
        leagueStats[prediction.league_type] = {
          total: 0,
          correct: 0,
          accuracy: 0,
        };
      }
      
      leagueStats[prediction.league_type].total++;
      if (prediction.is_correct) {
        leagueStats[prediction.league_type].correct++;
      }
    });

    Object.keys(leagueStats).forEach(league => {
      const stats = leagueStats[league];
      stats.accuracy = (stats.correct / stats.total * 100).toFixed(1);
    });

    return leagueStats;
  };

  const accuracyChartData = {
    labels: ["完全准确", "误差1内", "误差2内"],
    datasets: [
      {
        label: "预测数量",
        data: [
          getAccuracyStats()?.correct || 0,
          getAccuracyStats()?.withinOne || 0,
          getAccuracyStats()?.withinTwo || 0,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.2)",
          "rgba(255, 205, 86, 0.2)",
          "rgba(255, 99, 132, 0.2)",
        ],
        borderColor: [
          "rgb(75, 192, 192)",
          "rgb(255, 205, 86)",
          "rgb(255, 99, 132)",
        ],
        borderWidth: 1,
      },
    ],
  };

  const accuracyChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "预测准确性分析",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const leagueStats = getLeagueStats();
  const accuracyStats = getAccuracyStats();

  const columns = [
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
      render: (home, record) => (
        <Scoreboard
          home={record.home_team_zh || record.home_team}
          away={record.away_team_zh || record.away_team}
          homeGoals={home}
          awayGoals={record.predicted_away_goals}
          size="sm"
        />
      ),
    },
    {
      title: "实际比分",
      dataIndex: "actual_home_goals",
      key: "actual_home_goals",
      render: (home, record) => home !== null ? (
        <Scoreboard
          home={record.home_team_zh || record.home_team}
          away={record.away_team_zh || record.away_team}
          homeGoals={home}
          awayGoals={record.actual_away_goals}
          size="sm"
        />
      ) : (
        <span style={{ color: "#3D574A" }}>待定</span>
      ),
    },
    {
      title: "结果",
      dataIndex: "is_correct",
      key: "is_correct",
      render: (correct) => correct === null ? (
        <Tag>待复盘</Tag>
      ) : correct ? (
        <Tag color="success">正确</Tag>
      ) : (
        <Tag color="error">错误</Tag>
      ),
    },
    {
      title: "置信度",
      dataIndex: "confidence",
      key: "confidence",
      render: (confidence) => confidence != null ? `${(confidence * 100).toFixed(1)}%` : "N/A",
    },
    {
      title: "预测时间",
      dataIndex: "prediction_timestamp",
      key: "prediction_timestamp",
      render: (text) => new Date(text).toLocaleString(),
    },
    {
      title: "操作",
      key: "action",
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => showPredictionDetail(record)}
        >
          详情
        </Button>
      ),
    },
  ];

  return (
    <div>
      <h1>预测历史</h1>
      
      {/* 筛选器 */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Select
              placeholder="选择联赛"
              allowClear
              value={selectedLeague}
              onChange={setSelectedLeague}
              style={{ width: "100%" }}
            >
              <Option value="premier_league">英超</Option>
              <Option value="la_liga">西甲</Option>
              <Option value="bundesliga">德甲</Option>
              <Option value="ligue_1">法甲</Option>
              <Option value="serie_a">意甲</Option>
              <Option value="champions_league">欧冠</Option>
              <Option value="europa_league">欧联</Option>
              <Option value="world_cup">世界杯</Option>
              <Option value="european_championship">欧洲杯</Option>
            </Select>
          </Col>
          <Col span={8}>
            <RangePicker
              value={dateRange}
              onChange={setDateRange}
              style={{ width: "100%" }}
            />
          </Col>
          <Col span={6}>
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={fetchPredictions}
              style={{ width: "100%" }}
            >
              刷新数据
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 统计概览 */}
      {accuracyStats && (
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总预测数"
                value={accuracyStats.total}
                prefix={<TeamOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="正确预测"
                value={accuracyStats.correct}
                prefix={<TrophyOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="准确率"
                value={accuracyStats.accuracy}
                suffix="%"
                prefix={<BarChartOutlined />}
                valueStyle={{ 
                  color: accuracyStats.accuracy > 60 ? "#3f8600" : "#cf1322" 
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="误差1内"
                value={accuracyStats.withinOne}
                precision={0}
                prefix={<TeamOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 联赛统计 */}
      {Object.keys(leagueStats).length > 0 && (
        <Card title="联赛统计" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            {Object.entries(leagueStats).map(([league, stats]) => (
              <Col span={4} key={league}>
                <Card size="small">
                  <Statistic
                    title={league}
                    value={stats.accuracy}
                    suffix="%"
                    precision={1}
                    valueStyle={{ 
                      color: stats.accuracy > 60 ? "#3f8600" : "#cf1322" 
                    }}
                  />
                  <div style={{ fontSize: "12px", color: "#666" }}>
                    {stats.correct}/{stats.total}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      )}

      {/* 准确性图表 */}
      {accuracyStats && (
        <Card title="准确性分析" style={{ marginBottom: 16 }}>
          <div className="chart-container">
            <Line data={accuracyChartData} options={accuracyChartOptions} />
          </div>
        </Card>
      )}

      {/* 预测历史表格 */}
      <Card title="预测历史">
        <Table
          dataSource={filteredPredictions}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* 预测详情模态框 */}
      <Modal
        title="预测详情"
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedPrediction && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic
                  title="联赛"
                  value={selectedPrediction.league_type}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="预测时间"
                  value={new Date(selectedPrediction.prediction_timestamp).toLocaleString()}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="比赛时间"
                  value={selectedPrediction.match_date ? 
                    new Date(selectedPrediction.match_date).toLocaleString() : "待定"}
                />
              </Col>
            </Row>

            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Card title="比赛信息">
                  <p><strong>主队:</strong> {selectedPrediction.home_team}</p>
                  <p><strong>客队:</strong> {selectedPrediction.away_team}</p>
                  <p><strong>预测比分:</strong> <Scoreboard home={selectedPrediction.home_team} away={selectedPrediction.away_team} homeGoals={selectedPrediction.predicted_home_goals} awayGoals={selectedPrediction.predicted_away_goals} size="sm" /></p>
                  <p><strong>实际比分:</strong> {selectedPrediction.actual_home_goals !== null ? (
                    <Scoreboard home={selectedPrediction.home_team} away={selectedPrediction.away_team} homeGoals={selectedPrediction.actual_home_goals} awayGoals={selectedPrediction.actual_away_goals} size="sm" />
                  ) : "待定"}</p>
                  <p><strong>结果:</strong> {selectedPrediction.is_correct === null ? 
                    <Tag>待复盘</Tag> : selectedPrediction.is_correct ? 
                    <Tag color="success">正确</Tag> : 
                    <Tag color="error">错误</Tag>}</p>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="预测信息">
                  <p><strong>置信度:</strong> {selectedPrediction.confidence != null ? 
                    `${(selectedPrediction.confidence * 100).toFixed(1)}%` : "N/A"}</p>
                  <p><strong>预测主队进球:</strong> {selectedPrediction.predicted_home_goals}</p>
                  <p><strong>预测客队进球:</strong> {selectedPrediction.predicted_away_goals}</p>
                  <p><strong>误差:</strong> {selectedPrediction.actual_home_goals !== null ? 
                    `主队: ${Math.abs(selectedPrediction.predicted_home_goals - selectedPrediction.actual_home_goals)}, ` +
                    `客队: ${Math.abs(selectedPrediction.predicted_away_goals - selectedPrediction.actual_away_goals)}` : "待定"}</p>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default PredictionHistory;
