# A-Share Agent

A 股 AI Agent 模拟交易系统 — 基于多智能体协作的回测 + 纸上交易框架。

当前产品定位是面向短中期基金交易研究的辅助工具。系统使用股票或 ETF 的底层资产数据进行研究和模拟，不执行真实交易，也不把股票分析直接等同于基金专项分析。

## 技术栈

- **后端**：Python 3.11+ / FastAPI / LangGraph / DeepSeek 原生及 OpenAI-compatible LLM / AkShare / AnySearch / Serper / DDGS
- **前端**：React / Vite / TailwindCSS / shadcn/ui
- **依赖管理**：前端使用 pnpm，后端使用 uv 独立管理

## 核心功能

### 研究报告

每次完成一次研究分析后，系统会生成一份独立的 HTML 研究报告，并在前端的 **研究报告** 菜单中按报告维度集中展示。报告支持：

- 按标的、资产类型、来源和生成时间查看与搜索
- 在线预览和下载
- 查看关联对话作为追溯入口；对话不是报告列表的主维度
- 在报告元数据中记录生成时间、数据状态和联网搜索条数

报告内容由“数据上下文 + 多 Agent 分析 + 综合决策 + 固定报告模板”共同决定。固定模板负责报告结构，LLM 负责综合结论、风险、交易计划和各 Agent 观点；实时分析还会加入联网搜索结果，回测场景保持历史 as-of 数据边界，不主动查询实时网页。

报告文件保存到配置的 S3 兼容对象存储，报告索引和元数据保存到 ORM 配置的数据库。后端代理预览与下载请求，因此对象存储不需要公开读权限。

### 并行联网搜索

`search_web` 会并行查询已配置的服务和 DDGS，并合并去重：

- **AnySearch**：统一搜索 API，支持匿名访问；配置 `ANYSEARCH_API_KEY` 后会加入综合 `search_web`，也可以通过 `search_web_anysearch` 显式调用
- **Serper**：通过 Google Serper API 获取搜索结果，需要 `SERPER_API_KEY`，适合稳定的生产搜索
- **DDGS**：开源的 Python 元搜索库，不需要 API Key；它本身免费，但依赖的上游搜索引擎可能限流、阻断或调整服务，不能视为无限量、无条件可用的生产服务

如果没有配置 AnySearch 或 Serper，`search_web` 仍会使用 DDGS；需要明确指定供应商时，可调用 `search_web_anysearch` 或 `search_web_ddgs`。搜索结果会保留标题、摘要、来源和链接，并写入研究报告的“联网搜索结果”部分。

## 快速开始

### 前置要求

- Node.js >= 20.19
- pnpm >= 10
- Python >= 3.11
- uv (Python 包管理器)
- 至少一个可用的 LLM Provider API Key（DeepSeek 或 OpenAI-compatible）

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
# 编辑 .env，至少配置一个 LLM API Key
```

LLM 配置只从环境变量读取，不会写入数据库。使用 `LLM_PROVIDER` 选择 `deepseek` 或
`openai_compatible` 适配器，使用 `LLM_MODEL`、`LLM_TEMPERATURE` 和 `LLM_MAX_TOKENS`
配置模型参数；连接信息使用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。`OPENAI_BASE_URL`
默认是 `https://api.openai.com/v1`，使用其他 OpenAI-compatible 服务时在 `.env` 中修改它。
前端 **Settings** 页面只读展示当前生效配置。修改这些环境变量后需要重启后端。
适配器由 `LLM_PROVIDER` 显式决定，不根据模型名称推断。
生成研究报告时还需要配置 S3 兼容对象存储。

联网搜索配置：

```dotenv
# Serper；留空时 search_web 仍会并行使用 DDGS
SERPER_API_KEY=
SERPER_BASE_URL=https://google.serper.dev
SERPER_GL=cn
SERPER_HL=zh-cn

# AnySearch；API Key 可选。配置后会自动加入 search_web；不配置时仍可显式调用 search_web_anysearch 使用匿名额度
ANYSEARCH_API_KEY=
ANYSEARCH_BASE_URL=https://api.anysearch.com
ANYSEARCH_ZONE=cn
ANYSEARCH_LANGUAGE=zh-CN

# DDGS 元搜索参数
DDGS_REGION=cn-zh
DDGS_SAFESEARCH=moderate
```

综合搜索会并行查询已配置的 AnySearch、Serper 和 DDGS，再合并去重；未配置 AnySearch/Serper 时仍会使用 DDGS。也可以直接调用 `search_web_anysearch` 或 `search_web_ddgs` 指定搜索供应商。

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

Chat 页面支持股票、ETF 和 LOF 的行情查询、Agent Analysis 和 Backtest。用户可以直接在对话中描述标的、持有周期、风险约束或回测目标，Agent 会在对话里展示结构化分析、回测指标、资金曲线和研究报告。当前股票会接入基本面和新闻分析，场内基金主要使用实时/历史行情和技术分析。

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

任务默认关闭，paper 模式下所有订单只写入 ORM 管理的模拟账户，不连接真实券商。调度器会读取 A 股交易日历，数据源不可用时退化为周一至周五并记录警告。Portfolio 页面展示持仓、订单来源、日级净值快照和 WebSocket 事件；Chat 中的 Agent 决策可以在用户明确要求后提交到模拟账户。

实盘模式需要同时满足 `LIVE_TRADING_ENABLED=true`、自动化任务 `execution_mode=live` 且 `live_armed=true`、账户配置启用 `custom_http` Adapter。Adapter 对接一个由用户自行维护的交易网关：`POST /orders` 接收标准化订单，`DELETE /orders/{id}` 撤单，`GET /sync?account_id=...` 返回现金和持仓快照。默认配置和未实现的 provider 都会 fail-closed；未完成券商适配、风控和人工审批前，不应启用真实账户。

Chat 中的回测支持股票池，例如 `回测 000737,600519`。批量回测会让每个股票使用各自的历史 as-of 上下文，并合并到同一个组合中计算资金曲线和交易成本。

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
http://localhost:8000/api/health。`DATABASE_URL` 为空时，会话历史、任务状态和其他持久化数据
使用本地 SQLite，适合单节点部署；设置 `DATABASE_URL` 后，聊天数据、任务协调、事件日志和
账户、自动化、回测实验、制品索引、缓存、配置和 Agent checkpoint 统一使用 PostgreSQL，例如：
`postgres://postgres:password@localhost:5432/a_share_agent`。可通过 `FRONTEND_PORT` 和
`BACKEND_PORT` 修改映射端口。

### PostgreSQL 多节点部署

只有 PostgreSQL 模式支持多节点。多个 backend 节点可以同时运行，自动化调度、回测任务、
聊天任务领取、模拟盘事件和 Agent HITL checkpoint 都使用 PostgreSQL 的事务、行锁和租约。
节点异常后，过期租约可以由其他节点接管。

多节点部署要求：

- 所有节点使用同一个 `DATABASE_URL`，并连接同一个 PostgreSQL 数据库。
- 每个节点运行一个 backend worker；不需要也不建议在同一节点启动多个 Uvicorn worker。
- 前端请求和 SSE/WebSocket 可以被负载均衡到不同节点，任务和事件状态从 PostgreSQL 恢复。
- `DATABASE_PATH` 仅作为本地 SQLite 单节点模式的路径，不参与 PostgreSQL 多节点协调。

所有业务持久化模块均通过 ORM 访问数据库，SQLite 和 PostgreSQL 由同一套模型和 Repository
切换。多节点模式下，业务状态、任务协调、事件、缓存和配置都写入同一个 PostgreSQL 数据库。
制品二进制内容仍应配置共享的 S3 兼容对象存储；对象存储未配置时，不应使用节点本地目录承载
多节点制品文件。

SQLite 不承诺跨节点协调，也不建议通过共享文件系统把 SQLite 扩展成多节点部署。

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
│   │   ├── llm/             # Provider-neutral LLM 服务及 Provider 适配器
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
