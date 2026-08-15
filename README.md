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

### Agent 研究与交易

Chat 页面支持股票、ETF 和 LOF 的行情查询与研究。Analysis 页面可以选择标的类型，并填写预计持有天数、可投入资金、最大可接受亏损、当前仓位和持仓成本，供 Agent 生成更贴近短中期交易的决策。当前股票会接入基本面和新闻分析，场内基金主要使用实时/历史行情和技术分析。

对话会保存在当前浏览器本机，并支持从“历史”恢复；切换“新对话”可以开始新的会话。示例：

```text
分析 000737
为什么这样判断？
查看它最近的新闻
分析 ETF 510300
对比 ETF 510300 159915
```

场内基金使用 AkShare 的 `fund_etf_*` 或 `fund_lof_*` 接口；股票仍使用股票行情接口。所有分析和订单均为研究/模拟用途。

### LangSmith 追踪

在 `.env` 中配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后，Stock Agent 的路由及 LangGraph 分析流程会发送到 `LANGSMITH_PROJECT` 指定的项目。API Key 只通过环境变量读取，不会显示在前端。

### 无人值守 Agent 模拟交易

启动后打开前端的 **Agent 自动化** 页面，选择一个模拟账户并配置：

- 股票池、运行日和每日运行时间（默认 15:10，上海时区）
- `observe` 只记录决策、`confirm` 前端逐条确认、`auto` 自动生成模拟订单
- `next_open` 下一交易日开盘成交、`same_close` 当日收盘成交、`manual` 手动成交
- 单日亏损熔断、每次最大股票数和最大订单数

任务默认关闭，所有订单只写入本地 SQLite 模拟账户，不连接真实券商。调度器会读取 A 股交易日历，数据源不可用时退化为周一至周五并记录警告。Portfolio 页面展示持仓、订单来源、日级净值快照和 WebSocket 事件；Analysis 页面可以把单次 Agent 决策提交到模拟账户。

回测页面的股票代码支持逗号分隔的股票池，例如 `000737,600519`。批量回测会让每个股票使用各自的历史 as-of 上下文，并合并到同一个组合中计算资金曲线和交易成本。

自动化接口包括：

```text
GET  /api/automation/accounts/{account_id}
PUT  /api/automation/accounts/{account_id}
POST /api/automation/accounts/{account_id}/run
POST /api/automation/accounts/{account_id}/settle
GET  /api/automation/accounts/{account_id}/runs
GET  /api/automation/accounts/{account_id}/decisions
POST /api/automation/accounts/{account_id}/decisions/{decision_id}/confirm
```

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
