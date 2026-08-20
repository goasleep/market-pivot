"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from api.routers import artifacts, automation, backtest, chat, config, health, market, portfolio
from application.automation import automation_scheduler
from application.backtest_jobs import backtest_jobs
from application.chat_service import chat_store, chat_task_manager
from data.settings_store import load_llm_config
from data.tortoise_db import close_database, init_database


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_database()
    await load_llm_config()
    await chat_store.init()
    await chat_task_manager.start_worker()
    await backtest_jobs.start()
    await automation_scheduler.start()
    try:
        yield
    finally:
        await automation_scheduler.stop()
        await backtest_jobs.stop()
        await chat_task_manager.stop_worker()
        await chat_store.close()
        await close_database()

app = FastAPI(
    title="A-Share Agent API",
    description="AI Agent 驱动的 A 股模拟交易系统",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(automation.router, prefix="/api/automation", tags=["automation"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(market.router, prefix="/api/market", tags=["market"])


@app.get("/api/strategies", tags=["strategies"])
async def list_strategies():
    """List all available trading strategies."""
    from strategies.skill_manager import list_strategies as _list

    return {"strategies": await asyncio.to_thread(_list)}


@app.get("/api/system/status", tags=["system"])
async def system_status():
    """Get system status including circuit breaker states."""
    from data.akshare_provider import get_breaker_status

    return {"circuit_breakers": await asyncio.to_thread(get_breaker_status)}
