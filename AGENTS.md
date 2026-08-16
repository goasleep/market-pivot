# Repository Guidelines

## Project Structure & Module Organization

This pnpm monorepo contains an A-share AI trading simulation system.

## Product Positioning & Investor Profile

- Product positioning: a research and paper-trading assistant for a retail investor's short- to medium-term fund trading.
- User profile: a small retail investor who trades funds only and does not pursue long-term buy-and-hold investing.
- Analysis priorities: trend and market timing, entry/exit conditions, position sizing, stop-loss/take-profit, drawdown control, liquidity, fees, and holding-period risk.
- Do not frame outputs as long-term stock value-investing advice, guaranteed returns, or real order execution.
- Current implementation boundary: the analysis workflow is still stock-oriented and consumes six-digit A-share stock data. Treat stock analysis as underlying-asset research; do not present a stock recommendation as a fund recommendation or imply fund-specific analysis when fund data is unavailable.

- `backend/src/api/` contains FastAPI routes; `agents/` contains roles; `data/` wraps AkShare and caching; `engine/` contains backtesting and paper trading; `graph/` defines LangGraph workflows; `models/` contains schemas; and `llm/` contains the DeepSeek adapter.
- `backend/src/agents/stock_agent.py` routes conversational stock tasks such as analysis, quotes, history, news, strategies, and follow-up questions.
- `frontend/src/` contains the React application, organized into `pages/`, `components/`, `api/`, `types/`, and `router/`.
- Runtime/cache data is under `backend/data/`. Do not commit secrets, generated caches, or `__pycache__/` files.

## Build, Test, and Development Commands

From the repository root:

```bash
pnpm install                         # Install JavaScript dependencies
cd backend && uv sync              # Install Python dependencies
pnpm dev                             # Run frontend and backend together
pnpm dev:frontend                    # Run Vite at port 5173
pnpm dev:backend                     # Run FastAPI with reload at port 8000
pnpm build:frontend                  # Type-check and build the frontend
pnpm lint                            # Run frontend ESLint
pnpm format                          # Format frontend TypeScript/TSX
cd backend && uv run ruff check src
cd backend && uv run pytest
make setup                           # Initialize .env and install pnpm/uv dependencies
make up                              # Build Docker images and start both services
make down                            # Stop Docker Compose services
```

Copy `.env.example` to `.env` and provide the DeepSeek settings before running the backend. Keep API keys local.

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace Stock Agent and LangGraph runs. Use a project name in `LANGSMITH_PROJECT`; never commit the key. Persistent cache and application settings share the SQLite database configured by `DATABASE_PATH`.

For a containerized environment, use `make up`; Compose builds `backend/Dockerfile` and `frontend/Dockerfile`, persists backend data from `backend/data/`, and proxies frontend `/api` requests to the backend service.

## Coding Style & Naming Conventions

Use 2 spaces for TypeScript/TSX and the existing functional React style. Use `PascalCase` for components, `camelCase` for variables/hooks/API functions, 4 spaces for Python, and `snake_case` for Python modules/functions/variables. Use `PascalCase` for classes and Pydantic models. Keep Python lines within Ruff’s 120-character limit.

## Testing Guidelines

No tests are currently checked in. Add backend tests under `backend/tests/` using `pytest` (and `pytest-asyncio` for async endpoints), named like `test_backtester.py` or `test_<behavior>`. Frontend changes should pass the TypeScript build and lint commands; add focused tests for non-trivial UI or state logic.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so no project-specific convention can be verified. Use short, imperative subjects such as `Add portfolio risk limits` or `Fix chat stream handling`. Keep commits focused. Pull requests should explain the change, list validation commands, link an issue when relevant, and include screenshots for visible frontend changes. Call out configuration, API, cache, or schema changes.

### Commit Requirements

- Write a short, imperative commit subject in English, normally no longer than 72 characters, without a trailing period.
- Keep one logical feature or fix per commit; do not include unrelated user changes, generated files, secrets, caches, or `__pycache__/` files.
- Before committing, run the checks relevant to the changed areas. For backend changes, run `uv run ruff check src tests` and `uv run pytest`; for frontend changes, run `pnpm build:frontend` and `pnpm lint`.
- If the change affects both backend and frontend, run both sets of checks and verify the final diff with `git diff --check`.
- The commit body is optional, but use it when the subject cannot explain the behavior change or when configuration, API, cache, or schema impact needs to be recorded.
- When handing off a commit, report the commit hash, summarize the included changes, list validation commands, and explicitly mention intentionally excluded worktree changes.
- Run `make init` on a new checkout to install dependencies and configure the versioned hooks under `.githooks/`.
- The `commit-msg` hook enforces the Conventional Commits subject format and basic subject limits; the `pre-commit` hook runs staged-file checks for backend and frontend changes. Hooks may be bypassed with `--no-verify` only when the reason is documented in the handoff.

### Git Hook Workflow

- `make init` is the standard first-run command. It runs project setup and configures `git config core.hooksPath .githooks`.
- Use one logical change per commit. Valid commit types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`, `perf`, `style`, and `revert`.
- Commit subjects must follow `<type>: <imperative summary>` or `<type>(<scope>): <imperative summary>`, be no longer than 72 characters, and not end with a period.
- The `pre-commit` hook always runs `git diff --cached --check`. Staged backend changes additionally run `cd backend && uv run ruff check src tests` and `uv run pytest`; staged frontend changes additionally run `pnpm build:frontend` and `pnpm lint`.
- Do not use `--no-verify` to avoid a failing check during normal development. If an exceptional bypass is necessary, record the reason in the handoff and run the skipped checks manually before merging.

## Security & Configuration Tips

Treat DeepSeek credentials and cached market data as local configuration. Never hard-code keys or commit `.env` files. Changes to trading, backtesting, or external data access should preserve the research-only disclaimer and safely handle missing or stale data.
