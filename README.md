# IAZero — IA do Zero (local-first)

Este repositÃ³rio prepara a base para treinar e executar modelos **sem depender de APIs externas**.
Arquitetura em Python + PyTorch (CPU por padrÃ£o), com suporte a boa engenharia (lint/tests/notebooks).

## Estrutura
- `src/iac_core`: cÃ³digo-fonte da IA (data, models, utils, pipelines)
- `data/`: dados brutos e processados (nÃ£o versionados)
- `models/`: artefatos e checkpoints
- `notebooks/`: exploraÃ§Ã£o e P&D
- `scripts/`: utilitÃ¡rios de CLI
- `tests/`: testes unitÃ¡rios

## Como usar
1. Ative a venv e instale deps:
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
