import React, { useState, useEffect } from "react";
import { Card, Form, Input, Button, Switch, Select, message, Row, Col, Divider, Alert } from "antd";
import { request } from "../api";
import { UserOutlined, LockOutlined, MailOutlined } from "@ant-design/icons";

const { Option } = Select;

const Settings = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [userSettings, setUserSettings] = useState({
    username: "用户名",
    email: "user@example.com",
    notifications: true,
    darkMode: false,
    language: "zh-CN",
    timezone: "Asia/Shanghai",
  });

  useEffect(() => {
    fetchUserSettings();
    fetchApiSettings();
  }, []);

  const fetchApiSettings = async () => {
    try {
      const response = await request("/api/v1/user/api-settings");
      if (response.ok) {
        const data = await response.json();
        form.setFieldsValue({
          api_key: data.apiKey,
          rate_limit: String(data.rateLimit),
          enable_api: data.enabled,
        });
      }
    } catch (error) {
      console.error("获取API设置失败:", error);
    }
  };

  const fetchUserSettings = async () => {
    try {
      const response = await request("/api/v1/user/settings");
      if (response.ok) {
        const data = await response.json();
        setUserSettings(data);
        form.setFieldsValue(data);
      }
    } catch (error) {
      console.error("获取用户设置失败:", error);
    }
  };

  const handleSaveSettings = async (values) => {
    setLoading(true);
    try {
      const response = await request("/api/v1/user/settings", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        message.success("设置保存成功");
        setUserSettings(values);
      } else {
        const error = await response.json();
        message.error(error.error || "保存失败");
      }
    } catch (error) {
      message.error("网络错误，请重试");
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChange = async (values) => {
    setLoading(true);
    try {
      const response = await request("/api/v1/user/password", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        message.success("密码修改成功");
        form.resetFields(["current_password", "new_password", "confirm_password"]);
      } else {
        const error = await response.json();
        message.error(error.error || "密码修改失败");
      }
    } catch (error) {
      message.error("网络错误，请重试");
    } finally {
      setLoading(false);
    }
  };

  const handleApiSettings = async (values) => {
    setLoading(true);
    try {
      // 前端字段名 → 后端契约(apiKey/rateLimit/enabled)
      const payload = {
        apiKey: values.api_key || "",
        rateLimit: Number(values.rate_limit) || 1000,
        enabled: !!values.enable_api,
      };
      const response = await request("/api/v1/user/api-settings", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        message.success("API设置保存成功");
      } else {
        const error = await response.json();
        message.error(error.error || "保存失败");
      }
    } catch (error) {
      message.error("网络错误，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="settings-container">
      <h1>系统设置</h1>
      
      <Row gutter={[16, 16]}>
        <Col span={16}>
          {/* 基础设置 */}
          <Card title="基础设置" style={{ marginBottom: 16 }}>
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSaveSettings}
            >
              <Form.Item
                name="username"
                label="用户名"
                rules={[{ required: true, message: "请输入用户名" }]}
              >
                <Input prefix={<UserOutlined />} />
              </Form.Item>

              <Form.Item
                name="email"
                label="邮箱"
                rules={[
                  { required: true, message: "请输入邮箱" },
                  { type: "email", message: "请输入有效的邮箱地址" }
                ]}
              >
                <Input prefix={<MailOutlined />} />
              </Form.Item>

              <Form.Item
                name="language"
                label="语言"
                initialValue="zh-CN"
              >
                <Select>
                  <Option value="zh-CN">简体中文</Option>
                  <Option value="zh-TW">繁体中文</Option>
                  <Option value="en-US">English</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="timezone"
                label="时区"
                initialValue="Asia/Shanghai"
              >
                <Select>
                  <Option value="Asia/Shanghai">上海 (UTC+8)</Option>
                  <Option value="Asia/Tokyo">东京 (UTC+9)</Option>
                  <Option value="Asia/Seoul">首尔 (UTC+9)</Option>
                  <Option value="Europe/London">伦敦 (UTC+0)</Option>
                  <Option value="Europe/Paris">巴黎 (UTC+1)</Option>
                  <Option value="America/New_York">纽约 (UTC-5)</Option>
                  <Option value="America/Los_Angeles">洛杉矶 (UTC-8)</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="notifications"
                label="接收通知"
                valuePropName="checked"
                initialValue={true}
              >
                <Switch checkedChildren="开启" unCheckedChildren="关闭" />
              </Form.Item>

              <Form.Item
                name="darkMode"
                label="深色模式"
                valuePropName="checked"
                initialValue={false}
              >
                <Switch checkedChildren="开启" unCheckedChildren="关闭" />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  style={{ width: "100%" }}
                >
                  保存设置
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* 安全设置 */}
          <Card title="安全设置" style={{ marginBottom: 16 }}>
            <Form
              layout="vertical"
              onFinish={handlePasswordChange}
            >
              <Form.Item
                name="current_password"
                label="当前密码"
                rules={[{ required: true, message: "请输入当前密码" }]}
              >
                <Input.Password prefix={<LockOutlined />} />
              </Form.Item>

              <Form.Item
                name="new_password"
                label="新密码"
                rules={[
                  { required: true, message: "请输入新密码" },
                  { min: 6, message: "密码长度至少6位" }
                ]}
              >
                <Input.Password prefix={<LockOutlined />} />
              </Form.Item>

              <Form.Item
                name="confirm_password"
                label="确认密码"
                dependencies={["new_password"]}
                rules={[
                  { required: true, message: "请确认密码" },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue("new_password") === value) {
                        return Promise.resolve();
                      }
                      return Promise.reject(new Error("两次输入的密码不一致"));
                    },
                  }),
                ]}
              >
                <Input.Password prefix={<LockOutlined />} />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  style={{ width: "100%" }}
                >
                  修改密码
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* API设置 */}
          <Card title="API设置" style={{ marginBottom: 16 }}>
            <Form
              layout="vertical"
              onFinish={handleApiSettings}
            >
              <Form.Item
                name="api_key"
                label="API密钥"
              >
                <Input placeholder="输入API密钥(留空则关闭)" />
              </Form.Item>

              <Form.Item
                name="rate_limit"
                label="API调用频率限制"
                initialValue="1000"
              >
                <Input type="number" min="100" max="10000" />
              </Form.Item>

              <Form.Item
                name="enable_api"
                label="启用API访问"
                valuePropName="checked"
                initialValue={false}
              >
                <Switch checkedChildren="开启" unCheckedChildren="关闭" />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  style={{ width: "100%" }}
                >
                  保存API设置
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={8}>
          {/* 系统信息 */}
          <Card title="系统信息">
            <div style={{ marginBottom: 16 }}>
              <p><strong>版本:</strong> v1.0.0</p>
              <p><strong>数据库:</strong> PostgreSQL / SQLite</p>
              <p><strong>缓存:</strong> 未使用(进程内限流)</p>
              <p><strong>Python:</strong> 3.11</p>
            </div>
            <Divider />
            <div style={{ marginBottom: 16 }}>
              <p><strong>当前用户:</strong> {userSettings.username}</p>
              <p><strong>邮箱:</strong> {userSettings.email}</p>
              <p><strong>语言:</strong> {userSettings.language}</p>
              <p><strong>时区:</strong> {userSettings.timezone}</p>
            </div>
            <Divider />
            <div>
              <p><strong>通知:</strong> {userSettings.notifications ? "开启" : "关闭"}</p>
              <p><strong>深色模式:</strong> {userSettings.darkMode ? "开启" : "关闭"}</p>
            </div>
          </Card>

          {/* 帮助信息 */}
          <Card title="帮助信息" style={{ marginTop: 16 }}>
            <Alert
              message="使用提示"
              description="如果您在使用过程中遇到问题，请查看帮助文档或联系技术支持。"
              type="info"
              showIcon
            />
            <div style={{ marginTop: 16 }}>
              <p><strong>训练数据:</strong> 来源于数据库 matches 表(真实数据,由数据采集管道入库)</p>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Settings;
