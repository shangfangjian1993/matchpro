import React, { useState, useEffect } from "react";
import { Card, Row, Col, Statistic, Table, Tag, Select, DatePicker, Button } from "antd";
import { BarChartOutlined, TrophyOutlined, SettingOutlined, TeamOutlined } from "@ant-design/icons";
import { Line } from "react-chartjs-2";
import { request } from "../api";
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

const ModelPerformance = () => {
  const [loading, setLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [selectedLeague, setSelectedLeague] = useState(null);
  const [dateRange, setDateRange] = useState(null);

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  useEffect(() => {
    filterData();
  }, [performanceData, selectedLeague, dateRange]);

  const fetchPerformanceData = async () => {
    setLoading(true);
    try {
      const response = await request("/api/v1/models/performance");
      if (response.ok) {
        const data = await response.json();
        setPerformanceData(data);
      }
    } catch (error) {
      console.error("获取性能数据失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const filterData = () => {
    let filtered = [...performanceData];

    if (selectedLeague) {
      filtered = filtered.filter(item => item.league_type === selectedLeague);
    }

    if (dateRange) {
      const startDate = new Date(dateRange[0]);
      const endDate = new Date(dateRange[1]);
      filtered = filtered.filter(item => {
        const itemDate = new Date(item.training_date);
        return itemDate >= startDate && itemDate <= endDate;
      });
    }

    setFilteredData(filtered);
  };

  const performanceChartData = {
    labels: filteredData.map((item, index) => `${item.league_type} ${item.model_version}`),
    datasets: [
      {
        label: "准确率",
        data: filteredData.map(item => item.exact_accuracy * 100),
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
      {
        label: "Poisson损失",
        data: filteredData.map(item => item.poisson_loss),
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
      },
      {
        label: "MAE",
        data: filteredData.map(item => item.mae),
        borderColor: "rgb(255, 205, 86)",
        backgroundColor: "rgba(255, 205, 86, 0.2)",
      },
    ],
  };

  const performanceChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "模型性能对比",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const leagueAccuracyData = {};
  performanceData.forEach(item => {
    if (!leagueAccuracyData[item.league_type]) {
      leagueAccuracyData[item.league_type] = [];
    }
    leagueAccuracyData[item.league_type].push(item.exact_accuracy);
  });

  const leagueAccuracyChartData = {
    labels: Object.keys(leagueAccuracyData),
    datasets: [
      {
        label: "平均准确率",
        data: Object.values(leagueAccuracyData).map(accuracies => 
          accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length * 100
        ),
        backgroundColor: [
          "rgba(255, 99, 132, 0.2)",
          "rgba(54, 162, 235, 0.2)",
          "rgba(255, 205, 86, 0.2)",
          "rgba(75, 192, 192, 0.2)",
          "rgba(153, 102, 255, 0.2)",
        ],
        borderColor: [
          "rgb(255, 99, 132)",
          "rgb(54, 162, 235)",
          "rgb(255, 205, 86)",
          "rgb(75, 192, 192)",
          "rgb(153, 102, 255)",
        ],
        borderWidth: 1,
      },
    ],
  };

  const leagueAccuracyChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "各联赛模型准确率",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  };

  const performanceColumns = [
    {
      title: "联赛",
      dataIndex: "league_type",
      key: "league_type",
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: "模型版本",
      dataIndex: "model_version",
      key: "model_version",
    },
    {
      title: "MSE",
      dataIndex: "mse",
      key: "mse",
      render: (value) => value.toFixed(4),
    },
    {
      title: "MAE",
      dataIndex: "mae",
      key: "mae",
      render: (value) => value.toFixed(4),
    },
    {
      title: "RMSE",
      dataIndex: "rmse",
      key: "rmse",
      render: (value) => value.toFixed(4),
    },
    {
      title: "Poisson损失",
      dataIndex: "poisson_loss",
      key: "poisson_loss",
      render: (value) => value.toFixed(4),
    },
    {
      title: "准确率",
      dataIndex: "exact_accuracy",
      key: "exact_accuracy",
      render: (value) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: "训练日期",
      dataIndex: "training_date",
      key: "training_date",
      render: (text) => new Date(text).toLocaleString(),
    },
  ];

  const getBestModel = () => {
    if (filteredData.length === 0) return null;
    return filteredData.reduce((best, current) => 
      current.exact_accuracy > best.exact_accuracy ? current : best
    );
  };

  const getAverageMetrics = () => {
    if (filteredData.length === 0) return null;
    
    const sum = filteredData.reduce((acc, item) => ({
      mse: acc.mse + item.mse,
      mae: acc.mae + item.mae,
      rmse: acc.rmse + item.rmse,
      poisson_loss: acc.poisson_loss + item.poisson_loss,
      exact_accuracy: acc.exact_accuracy + item.exact_accuracy,
    }), {
      mse: 0,
      mae: 0,
      rmse: 0,
      poisson_loss: 0,
      exact_accuracy: 0,
    });

    const count = filteredData.length;
    
    return {
      mse: sum.mse / count,
      mae: sum.mae / count,
      rmse: sum.rmse / count,
      poisson_loss: sum.poisson_loss / count,
      exact_accuracy: sum.exact_accuracy / count,
    };
  };

  if (loading) {
    return <div className="loading-spinner">加载中...</div>;
  }

  const bestModel = getBestModel();
  const averageMetrics = getAverageMetrics();

  return (
    <div>
      <h1>模型性能监控</h1>
      
      {/* 筛选器 */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
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
          <Col span={8}>
            <Button
              type="primary"
              onClick={fetchPerformanceData}
              style={{ width: "100%" }}
            >
              刷新数据
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 关键指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="模型总数"
              value={filteredData.length}
              prefix={<SettingOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均准确率"
              value={averageMetrics ? averageMetrics.exact_accuracy * 100 : 0}
              precision={1}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ 
                color: averageMetrics && averageMetrics.exact_accuracy > 0.6 ? "#3f8600" : "#cf1322" 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均Poisson损失"
              value={averageMetrics ? averageMetrics.poisson_loss : 0}
              precision={3}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均MAE"
              value={averageMetrics ? averageMetrics.mae : 0}
              precision={3}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 最佳模型 */}
      {bestModel && (
        <Card title="最佳模型" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic
                title="联赛"
                value={bestModel.league_type}
                prefix={<TrophyOutlined />}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="准确率"
                value={bestModel.exact_accuracy * 100}
                precision={1}
                suffix="%"
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Poisson损失"
                value={bestModel.poisson_loss}
                precision={3}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* 图表 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col span={16}>
          <Card title="性能对比">
            <div className="chart-container">
              <Line data={performanceChartData} options={performanceChartOptions} />
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="联赛准确率">
            <div className="chart-container">
              <Line data={leagueAccuracyChartData} options={leagueAccuracyChartOptions} />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 性能数据表格 */}
      <Card title="详细性能数据">
        <Table
          dataSource={filteredData}
          columns={performanceColumns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: 1000 }}
        />
      </Card>
    </div>
  );
};

export default ModelPerformance;
