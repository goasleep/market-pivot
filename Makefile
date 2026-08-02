.PHONY: help install setup dev build up start down restart logs docker-build clean

help:
	@echo "Available commands:"
	@echo "  make setup        Initialize .env and install local dependencies"
	@echo "  make install      Install frontend (pnpm) and backend (uv) dependencies"
	@echo "  make dev          Start local development servers"
	@echo "  make build        Build the frontend locally"
	@echo "  make up           Build images and start Docker Compose in background"
	@echo "  make start        Alias for make up"
	@echo "  make docker-build Build Docker images without starting services"
	@echo "  make down         Stop and remove Compose services"
	@echo "  make restart      Restart Compose services"
	@echo "  make logs         Follow Compose service logs"

setup:
	@test -f .env || cp .env.example .env
	$(MAKE) install

install:
	pnpm install
	cd backend && uv sync

dev:
	pnpm dev

build:
	pnpm build:frontend

docker-build:
	docker compose build

up:
	docker compose up --build -d

start: up

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build -d

logs:
	docker compose logs -f

clean:
	docker compose down --rmi local --volumes --remove-orphans
