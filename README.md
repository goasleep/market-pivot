# A-Share Agent

A 股 AI Agent 模拟交易系统 — 基于多智能体协作的回测 + 纸上交易框架。

当前产品定位是面向短中期基金交易研究的辅助工具。系统使用股票或 ETF 的底层资产数据进行研究和模拟，不执行真实交易，也不把股票分析直接等同于基金专项分析。

## 技术栈

- **后端**：Python 3.11+ / FastAPI / LangGraph / DeepSeek API / AkShare / Serper / DDGS
- **前端**：React / Vite / TailwindCSS / shadcn/ui
- **Monorepo**：pnpm workspaces

## 核心功能

### 研究报告

每次完成一次研究分析后，系统会生成一份独立的 HTML 研究报告，并在前端的 **研究报告** 菜单中按报告维度集中展示。报告支持：

- 按标的、资产类型、来源和生成时间查看与搜索
- 在线预览和下载
- 查看关联对话作为追溯入口；对话不是报告列表的主维度
- 在报告元数据中记录生成时间、数据状态和联网搜索条数

报告内容由“数据上下文 + 多 Agent 分析 + 综合决策 + 固定报告模板”共同决定。固定模板负责报告结构，LLM 负责综合结论、风险、交易计划和各 Agent 观点；实时分析还会加入联网搜索结果，回测场景保持历史 as-of 数据边界，不主动查询实时网页。

报告文件保存到配置的 S3 兼容对象存储，报告索引和元数据保存到 SQLite。后端代理预览与下载请求，因此对象存储不需要公开读权限。

### 并行联网搜索

`search_web` 会并行查询两个来源并合并去重：

- **Serper**：通过 Google Serper API 获取搜索结果，需要 `SERPER_API_KEY`，适合稳定的生产搜索
- **DDGS**：开源的 Python 元搜索库，不需要 API Key；它本身免费，但依赖的上游搜索引擎可能限流、阻断或调整服务，不能视为无限量、无条件可用的生产服务

如果没有配置 Serper，`search_web` 仍会使用 DDGS；如果需要明确只使用 DDGS，可调用 `search_web_ddgs`。搜索结果会保留标题、摘要、来源和链接，并写入研究报告的“联网搜索结果”部分。

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
# 编辑 .env 填入 DeepSeek API Key；生成研究报告时还需要 S3 兼容对象存储配置
```

联网搜索配置：

```dotenv
# Serper；留空时 search_web 仍会并行使用 DDGS
SERPER_API_KEY=
SERPER_BASE_URL=https://google.serper.dev
SERPER_GL=cn
SERPER_HL=zh-cn

# DDGS 元搜索参数
DDGS_REGION=cn-zh
DDGS_SAFESEARCH=moderate
```

综合搜索会并行查询 Serper 和 DDGS，再合并去重；未配置 Serper 时仍会使用 DDGS。也可以直接调用 `search_web_ddgs` 使用 DDGS 元搜索。

产物报告默认使用 S3 兼容对象存储，不写入后端本地磁盘。支持 AWS S3、MinIO、Ceph、Cloudflare R2 等服务；通过以下配置项设置：

```dotenv
S3_ENDPOINT_URL=
S3_BUCKET=
S3_REGION=us-east-1
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_SESSION_TOKEN=
S3_ADDRESSING_STYLE=path
S3_ARTIFACTS_PREFIX=a-share-agent/artifacts
```

单元测试可以显式注入本地存储适配器；生产环境应配置 S3 兼容对象存储。

### 启动开发服务器

```bash
# 同时启动前后端（推荐）
make dev

# 也可以直接使用 pnpm
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

场内基金使用 AkShare 的 `fund_etf_*` 或 `fund_lof_*` 接口；股票仍使用股票行情接口。默认分析和订单均为研究/模拟用途；实盘需额外配置并通过独立安全门禁。

研究报告接口包括：

```text
GET /api/artifacts
GET /api/artifacts/{artifact_id}
GET /api/artifacts/{artifact_id}/preview
GET /api/artifacts/{artifact_id}/download
```

### LangSmith 追踪

在 `.env` 中配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后，Stock Agent 的路由及 LangGraph 分析流程会发送到 `LANGSMITH_PROJECT` 指定的项目。API Key 只通过环境变量读取，不会显示在前端。

### Langfuse 追踪

在 `.env` 中配置 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 和可选的 `LANGFUSE_BASE_URL` 后，Stock Agent 对话、研究分析及回测中的 LangGraph/LLM 调用会发送到 Langfuse。每次根调用都会创建独立 callback，并附带标的、日期、策略和会话元数据；未配置 key 时自动跳过。API Key 只通过环境变量读取，不会显示在前端。

### 无人值守 Agent 模拟交易

启动后打开前端的 **Agent 自动化** 页面，选择一个模拟账户并配置：

- 股票池、运行日和每日运行时间（默认 15:10，上海时区）
- `observe` 只记录决策、`confirm` 前端逐条确认、`auto` 自动生成模拟订单
- `next_open` 下一交易日开盘成交、`same_close` 当日收盘成交、`manual` 手动成交
- 单日亏损熔断、每次最大股票数和最大订单数

任务默认关闭，paper 模式下所有订单只写入本地 SQLite 模拟账户，不连接真实券商。调度器会读取 A 股交易日历，数据源不可用时退化为周一至周五并记录警告。Portfolio 页面展示持仓、订单来源、日级净值快照和 WebSocket 事件；Analysis 页面可以把单次 Agent 决策提交到模拟账户。

实盘模式需要同时满足 `LIVE_TRADING_ENABLED=true`、自动化任务 `execution_mode=live` 且 `live_armed=true`、账户配置启用 `custom_http` Adapter。Adapter 对接一个由用户自行维护的交易网关：`POST /orders` 接收标准化订单，`DELETE /orders/{id}` 撤单，`GET /sync?account_id=...` 返回现金和持仓快照。默认配置和未实现的 provider 都会 fail-closed；未完成券商适配、风控和人工审批前，不应启用真实账户。

回测页面的股票代码支持逗号分隔的股票池，例如 `000737,600519`。批量回测会让每个股票使用各自的历史 as-of 上下文，并合并到同一个组合中计算资金曲线和交易成本。

### 测试与构建

```bash
# 后端测试与静态检查
cd backend && uv run pytest
cd backend && uv run ruff check src

# 前端类型检查与生产构建
pnpm build:frontend
```

自动化接口包括：

```text
GET  /api/automation/accounts/{account_id}
PUT  /api/automation/accounts/{account_id}
POST /api/automation/accounts/{account_id}/run
POST /api/automation/accounts/{account_id}/settle
POST /api/automation/accounts/{account_id}/live-sync
GET  /api/automation/accounts/{account_id}/runs
GET  /api/automation/accounts/{account_id}/decisions
POST /api/automation/accounts/{account_id}/decisions/{decision_id}/confirm
PUT  /api/portfolio/accounts/{account_id}/live
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
http://localhost:8000/api/health。会话历史通过 Tortoise ORM 保存，默认使用
`DATABASE_PATH` 指定的 SQLite 文件；设置 `DATABASE_URL` 后可切换到 PostgreSQL，例如
`postgres://postgres:password@localhost:5432/a_share_agent`。行情缓存、配置和模拟盘等其他
现有模块仍使用 `DATABASE_PATH` 指定的本地 SQLite 文件。可通过 `FRONTEND_PORT` 和
`BACKEND_PORT` 修改映射端口。

## 项目结构

```
a-share-agent/
├── backend/                 # Python 后端
│   ├── src/
│   │   ├── api/             # FastAPI 路由
│   │   ├── agents/          # Agent 角色定义
│   │   ├── data/            # 数据源封装 (AkShare)
│   │   ├── artifacts/        # 研究报告生成、索引与对象存储
│   │   ├── application/      # 对话、研究与任务服务
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
