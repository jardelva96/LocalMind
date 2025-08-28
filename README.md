# IAZero — IA do Zero (local-first)

Este repositorio prepara a base para treinar e executar modelos **sem depender de APIs externas**.
Arquitetura em Python + PyTorch (CPU por padrão), com suporte a boa engenharia (lint/tests/notebooks).

## Estrutura
- `src/iac_core`: Codigo-fonte da IA (data, models, utils, pipelines)
- `data/`: dados brutos e processados (não versionados)
- `models/`: artefatos e checkpoints
- `notebooks/`: exploração e P&D
- `scripts/`: utilitarios de CLI
- `tests/`: testes unitarios

## Como usar
1. Ative a venv e instale deps:
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
