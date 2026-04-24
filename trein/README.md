# Pasta `trein/` — pipeline LoRA

- **`requirements.txt`** — dependências só desta pipeline (sem FastAPI/PostgreSQL). `pip install -r trein/requirements.txt` a partir da raiz do repositório.
- **`data_config.py`** — modelos, pastas `trein/outputs/`, `trein/data/`, versões de snapshot.
- **`train_lora.py`**, **`merge_lora.py`**, **`inferir.py`**, **`build_snapshot.py`**, **`verificar_ambiente.py`**
- **`data/`** — `raw/`, `snapshots/`, exemplos.
- **`prompt_system/`** — ficheiros de prompt usados no fluxo de dados, se aplicável.
- **`treina.sh`**, **`prompt.sh`** — atalhos (a partir da raiz do repositório).

Documentação longa: [`../README.md`](../README.md) (guia geral) e ficheiros `README_DATASETS.md`, `README_SNAPSHOT.md`, `README_INFERIR.md` **nesta pasta**.

Saídas de treino (adapter, merge): por defeito **`trein/outputs/lora_adapter`** e **`trein/outputs/merged_model`**.
