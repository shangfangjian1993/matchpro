import React, { useState, useEffect, useRef } from "react";
import { Form, Select, Input, Button, Card, Row, Col, Table, Tag, message, Switch, Progress, Modal } from "antd";
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

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ModelTraining = () => {
  const [form] = Form.useForm();
  const [leagues, setLeagues] = useState([]);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [task, setTask] = useState(null);
  const [importOpen, setImportOpen] = useState(false);
  const [importLeague, setImportLeague] = useState(null);
  const [importJson, setImportJson] = useState("");
  const [importing, setImporting] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    fetchLeagues();
    fetchTrainingHistory();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
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

  const fetchTrainingHistory = async () => {
    try {
      const response = await request("/api/v1/models/performance");
      if (response.ok) {
        const data = await response.json();
        setTrainingHistory(data);
      }
    } catch (error) {
      console.error("获取训练历史失败:", error);
    }
  };

  const importTemplate = `[
  {"home_team": "Manchester City", "away_team": "Arsenal", "home_goals": 2, "away_goals": 1, "match_date": "2026-08-15T19:30:00", "home_xg": 2.3, "away_xg": 1.1},
  {"home_team": "Liverpool", "away_team": "Chelsea", "home_goals": 1, "away_goals": 1, "match_date": "2026-08-16T19:30:00"}
]`;

  const handleImport = async () => {
    if (!importLeague) {
      message.warning("请先选择目标联赛");
      return;
    }
    let matches;
    try {
      matches = JSON.parse(importJson);
    } catch (e) {
      message.error("JSON 格式错误: " + e.message);
      return;
    }
    if (!Array.isArray(matches) || matches.length === 0) {
      message.error("请输入非空比赛数组");
      return;
    }
    setImporting(true);
    try {
      const response = await request("/api/v1/matches/batch", {
        method: "POST",
        body: JSON.stringify({ league_type: importLeague, matches }),
      });
      const data = await response.json();
      if (response.ok) {
        message.success(data.message || "导入成功");
        setImportOpen(false);
        setImportJson("");
      } else {
        const err = data.error || {};
        const msg = typeof err === "string" ? err : (err.message || "导入失败");
        message.error(msg);
        if (err && Array.isArray(err.errors) && err.errors.length) {
          console.error("导入错误明细:", err.errors.slice(0, 5));
        }
      }
    } catch (e) {
      message.error("网络错误,请重试");
    } finally {
      setImporting(false);
    }
  };

  const onFinish = async (values) => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTask(null);

    try {
      const response = await request("/api/v1/training", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          league_type: values.league_type,
          target_column: values.target_column,
          cross_validation: values.cross_validation,
          cv_folds: values.cv_folds,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        message.error(data.error || "训练提交失败");
        setIsTraining(false);
        return;
      }
      setTask(data.task);
      setTrainingProgress(10);

      // 真实轮询训练任务状态(2 秒间隔)
      pollRef.current = setInterval(async () => {
        try {
          const r = await request(`/api/v1/training/${data.task.public_id || data.task.id}`);
          const t = await r.json();
          if (!r.ok) {
            clearInterval(pollRef.current);
            message.error(t.error || "查询训练状态失败");
            setIsTraining(false);
            return;
          }
          setTask(t);
          setTrainingProgress(
            t.status === "pending" ? 10
              : t.status === "running" ? 60
              : t.status === "succeeded" ? 100 : 0
          );
          if (t.status === "succeeded") {
            clearInterval(pollRef.current);
            message.success("模型训练完成");
            fetchTrainingHistory();
            setIsTraining(false);
          } else if (t.status === "failed") {
            clearInterval(pollRef.current);
            message.error(t.message || "训练失败");
            setIsTraining(false);
          }
        } catch (e) {
          clearInterval(pollRef.current);
          message.error("网络错误，请重试");
          setIsTraining(false);
        }
      }, 2000);
    } catch (error) {
      message.error("网络错误，请重试");
      setIsTraining(false);
    }
  };

  const trainingHistoryColumns = [
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

  const performanceChartData = {
    labels: trainingHistory.map((item, index) => `训练 ${index + 1}`),
    datasets: [
      {
        label: "准确率",
        data: trainingHistory.map(item => item.exact_accuracy * 100),
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
      {
        label: "Poisson损失",
        data: trainingHistory.map(item => item.poisson_loss),
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
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
        text: "模型性能趋势",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="model-training-form">
      <h1>模型训练</h1>
      
      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="新建模型训练" extra={<Button onClick={() => setImportOpen(true)}>导入比赛数据</Button>}>
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
                name="target_column"
                label="目标变量"
                initialValue="goals"
              >
                <Select>
                  <Option value="goals">进球数</Option>
                  <Option value="xg">预期进球</Option>
                  <Option value="shots">射门次数</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="test_split_ratio"
                label="测试集比例"
                initialValue={0.2}
              >
                <Input type="number" min="0.1" max="0.5" step="0.1" />
              </Form.Item>

              <Form.Item
                name="cross_validation"
                label="交叉验证"
                valuePropName="checked"
                initialValue={true}
              >
                <Switch checkedChildren="开启" unCheckedChildren="关闭" />
              </Form.Item>

              {form.getFieldValue("cross_validation") && (
                <Form.Item
                  name="cv_folds"
                  label="交叉验证折数"
                  initialValue={5}
                >
                  <Input type="number" min="3" max="10" />
                </Form.Item>
              )}

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={isTraining}
                  disabled={isTraining}
                  style={{ width: "100%" }}
                >
                  {isTraining ? "训练中..." : "开始训练"}
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={8}>
          <Card title="训练状态">
            {isTraining ? (
              <div>
                <Progress
                  percent={trainingProgress}
                  status={task && task.status === "failed" ? "exception" : "active"}
                />
                <div style={{ marginTop: 16 }}>
                  <p>{task ? task.message : "正在提交训练任务..."}</p>
                  <p>状态: {task ? task.status : "pending"}</p>
                </div>
              </div>
            ) : (
              <p>当前没有正在进行的训练</p>
            )}
          </Card>
        </Col>
      </Row>

      {trainingHistory.length > 0 && (
        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={16}>
            <Card title="训练历史">
              <Table
                dataSource={trainingHistory}
                columns={trainingHistoryColumns}
                rowKey="id"
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card title="性能趋势">
              <div className="chart-container">
                <Line data={performanceChartData} options={performanceChartOptions} />
              </div>
            </Card>
          </Col>
        </Row>
      )}
      <Modal
        title="批量导入比赛数据"
        open={importOpen}
        onCancel={() => setImportOpen(false)}
        onOk={handleImport}
        okText="导入"
        cancelText="取消"
        confirmLoading={importing}
        width={760}
      >
        <p style={{ marginBottom: 8 }}>选择目标联赛,然后粘贴比赛 JSON 数组(点击"填入示例"查看格式):</p>
        <Select
          style={{ width: "100%", marginBottom: 12 }}
          placeholder="选择目标联赛"
          value={importLeague}
          onChange={setImportLeague}
        >
          {leagues.map(league => (
            <Option key={league.id} value={league.league_type}>
              {league.name} ({league.season})
            </Option>
          ))}
        </Select>
        <Input.TextArea
          rows={8}
          value={importJson}
          onChange={(e) => setImportJson(e.target.value)}
          placeholder='[{"home_team": "Manchester City", "away_team": "Arsenal", "home_goals": 2, "away_goals": 1, "match_date": "2026-08-15T19:30:00"}]'
        />
        <Button size="small" style={{ marginTop: 8 }} onClick={() => setImportJson(importTemplate)}>
          填入示例
        </Button>
        <p style={{ marginTop: 8, color: "#888", fontSize: 12 }}>
          字段说明: home_team / away_team / home_goals / away_goals 必填; match_date 可选(ISO 格式);
          支持 xg、射门、控球、传球成功率等指标字段(详见 API 文档)。
        </p>
      </Modal>
    </div>
  );
};

export default ModelTraining;
