# A-Share Agent

A 股 AI Agent 模拟交易系统 — 基于多智能体协作的回测 + 纸上交易框架。

## 技术栈

- **后端**：Python 3.11+ / FastAPI / LangGraph / DeepSeek API / AkShare
- **前端**：React / Vite / TailwindCSS / shadcn/ui
- **Monorepo**：pnpm workspaces

## 快速开始

### 前置要求

- Node.js >= 20
- pnpm >= 9
- Python >= 3.11
- uv (Python 包管理器)
- DeepSeek API Key

### 安装

```bash
# 安装前端依赖
pnpm install

# 安装后端依赖
cd backend
uv sync
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

### 启动开发服务器

```bash
# 同时启动前后端
pnpm dev

# 或分别启动
pnpm dev:frontend   # 前端 -> http://localhost:5173
pnpm dev:backend    # 后端 -> http://localhost:8000
```

### Stock Agent

Chat 页面现在是面向 A 股任务的 Stock Agent，可处理单只股票的综合分析、实时行情、历史走势、新闻舆情和策略查询。它会记住当前会话中的最近股票代码，因此可以继续追问，例如：

```text
分析 000737
为什么这样判断？
查看它最近的新闻
```

### LangSmith 追踪

在 `.env` 中配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后，Stock Agent 的路由及 LangGraph 分析流程会发送到 `LANGSMITH_PROJECT` 指定的项目。API Key 只通过环境变量读取，不会显示在前端。

### 一键安装与启动

```bash
# 本地安装 pnpm + uv 依赖，并初始化 .env
make setup

# 使用 Docker 构建并后台启动前后端
make up

# 查看日志 / 停止服务
make logs
make down
```

Docker 启动后访问 http://localhost:5173，后端健康检查地址为
http://localhost:8000/api/health。DeepSeek 配置和运行数据统一由后端 SQLite 数据库管理；可通过
`FRONTEND_PORT` 和 `BACKEND_PORT` 修改映射端口。

## 项目结构

```
a-share-agent/
├── backend/                 # Python 后端
│   ├── src/
│   │   ├── api/             # FastAPI 路由
│   │   ├── agents/          # Agent 角色定义
│   │   ├── data/            # 数据源封装 (AkShare)
│   │   ├── llm/             # DeepSeek LLM 适配器
│   │   ├── engine/          # 回测 + 模拟交易引擎
│   │   ├── graph/           # LangGraph 工作流
│   │   └── models/          # 数据模型
│   └── pyproject.toml
├── frontend/                # React 前端
│   ├── src/
│   │   ├── components/      # UI 组件
│   │   ├── pages/           # 页面
│   │   ├── api/             # API 调用
│   │   ├── hooks/           # 自定义 Hooks
│   │   ├── types/           # TypeScript 类型
│   │   └── router/          # 路由
│   └── package.json
├── pnpm-workspace.yaml
└── package.json
```

## 免责声明

本项目仅供学习和研究目的，不构成任何投资建议。股市有风险，投资需谨慎。
