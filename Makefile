PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install install-backend install-frontend migrate backend frontend test build demo demo-fresh autoresearch compare clean-local

install: install-backend install-frontend

install-backend:
	python3 -m venv .venv
	$(PIP) install -r backend/requirements.txt

install-frontend:
	cd frontend && npm install

migrate:
	cd backend && ../$(PY) -m alembic upgrade head

backend:
	cd backend && ../$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && ../$(PY) -m pytest

build:
	cd frontend && npm run build

demo:
	$(PY) scripts/demo.py

demo-fresh:
	$(PY) scripts/demo.py --reset --compare

autoresearch:
	$(PY) scripts/autoresearch.py $(ARGS)

compare:
	curl -s -X POST localhost:8000/api/evaluation/compare -H 'Content-Type: application/json' -d '{"seeds":[0,1,2],"budget":6}'

clean-local:
	rm -rf data workspace
