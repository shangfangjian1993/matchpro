import React, { useState, useEffect } from "react";
import { Form, Select, Input, Button, Card, Row, Col, Statistic, Tag, message } from "antd";
import { BarChartOutlined, TeamOutlined } from "@ant-design/icons";
import { Pie } from "react-chartjs-2";
import { request } from "../api";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";

const { Option } = Select;

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend
);

const TournamentPrediction = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [leagues, setLeagues] = useState([]);
  const [teams, setTeams] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [allTeams, setAllTeams] = useState([]);

  useEffect(() => {
    fetchLeagues();
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

  // 从后端拉取该联赛的真实球队(matches 表去重)
  const fetchLeagueTeams = async (leagueType) => {
    try {
      const league = leagues.find(l => l.league_type === leagueType);
      if (!league) return;
      const response = await request(`/api/v1/leagues/${league.id}/teams`);
      if (response.ok) {
        const data = await response.json();
        setAllTeams(data.map(t => t.name));
      }
    } catch (error) {
      console.error("获取球队列表失败:", error);
    }
  };

  const handleLeagueChange = (value) => {
    form.setFieldsValue({ teams: [] });
    setTeams([]);
    setAllTeams([]);
    if (value) {
      fetchLeagueTeams(value);
    }
  };

  // antd multiple Select 的 onChange 直接传数组,直接受控更新
  const handleTeamsChange = (value) => {
    setTeams(value);
  };

  const removeTeam = (team) => {
    setTeams(teams.filter(t => t !== team));
  };

  const onFinish = async (values) => {
    if (teams.length < 2) {
      message.error("至少需要2支球队才能进行赛事预测");
      return;
    }

    setLoading(true);
    try {
      const response = await request("/api/v1/predictions/tournament", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          league_type: values.league_type,
          teams: teams,
          num_simulations: values.num_simulations || 1000,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
        message.success("赛事预测完成");
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

  const tournamentChartData = {
    labels: prediction?.champion_probabilities?.map(item => item.team) || [],
    datasets: [
      {
        label: "冠军概率",
        data: prediction?.champion_probabilities?.map(item => item.probability * 100) || [],
        backgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#9966FF",
          "#FF9F40",
          "#FF6384",
          "#C9CBCF",
        ],
        borderWidth: 1,
      },
    ],
  };

  const tournamentChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "right",
      },
      title: {
        display: true,
        text: "冠军概率分布",
      },
    },
  };

  return (
    <div className="tournament-form">
      <h1>赛事预测</h1>
      
      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="新建赛事预测">
            <Form
              form={form}
              layout="vertical"
              onFinish={onFinish}
            >
              <Form.Item
                name="league_type"
                label="赛事类型"
                rules={[{ required: true, message: "请选择赛事类型" }]}
              >
                <Select placeholder="选择赛事类型" onChange={handleLeagueChange}>
                  {leagues.map(league => (
                    <Option key={league.id} value={league.league_type}>
                      {league.name} ({league.season})
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                label="参赛球队"
              >
                <Select
                  mode="tags"
                  style={{ width: "100%" }}
                  placeholder="选择或输入球队名称(需与库中英文规范名一致)"
                  value={teams}
                  onChange={handleTeamsChange}
                >
                  {allTeams
                    .filter(team => !teams.includes(team))
                    .map(team => (
                      <Option key={team} value={team}>
                        {team}
                      </Option>
                    ))}
                </Select>
                <div style={{ marginTop: 8 }}>
                  {teams.map(team => (
                    <Tag
                      key={team}
                      closable
                      onClose={() => removeTeam(team)}
                      style={{ marginBottom: 4 }}
                    >
                      {team}
                    </Tag>
                  ))}
                </div>
              </Form.Item>

              <Form.Item
                name="num_simulations"
                label="模拟次数"
                initialValue={1000}
                rules={[{ required: true, message: "请输入模拟次数" }]}
              >
                <Input type="number" min="100" max="10000" />
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
          <Card title="当前参赛球队">
            {teams.length > 0 ? (
              <div>
                {teams.map((team, index) => (
                  <div key={team} style={{ marginBottom: 8 }}>
                    <TeamOutlined style={{ marginRight: 8 }} />
                    {team}
                  </div>
                ))}
              </div>
            ) : (
              <p>请选择参赛球队</p>
            )}
          </Card>
        </Col>
      </Row>

      {prediction && (
        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col span={12}>
            <Card title="冠军概率">
              <div className="chart-container">
                <Pie data={tournamentChartData} options={tournamentChartOptions} />
              </div>
            </Card>
          </Col>

          <Col span={12}>
            <Card title="预测结果">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Statistic
                    title="总模拟次数"
                    value={prediction.num_simulations}
                    prefix={<BarChartOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="参赛球队数"
                    value={prediction.champion_probabilities?.length || 0}
                    prefix={<TeamOutlined />}
                  />
                </Col>
              </Row>

              <div style={{ marginTop: 16 }}>
                <h3>冠军概率排行</h3>
                {prediction.champion_probabilities?.slice(0, 5).map((item, index) => (
                  <div key={item.team} style={{ marginBottom: 8 }}>
                    <strong>{index + 1}. {item.team}</strong>: {(item.probability * 100).toFixed(1)}%
                  </div>
                ))}
              </div>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default TournamentPrediction;
