# Pasta `trein/` — pipeline LoRA

- `**requirements.txt**` — dependências só desta pipeline (sem FastAPI/PostgreSQL). `pip install -r trein/requirements.txt` a partir da raiz do repositório.
- `**data_config.py**` — modelos, pastas `trein/outputs/`, `trein/data/`, versões de snapshot.
- `**train_lora.py**`, `**merge_lora.py**`, `**build_snapshot.py**`, `**verificar_ambiente.py**` (e opcionalmente `**inferir.py**` — teste de uma geração; o fluxo “real” de treino é `train_lora` + `merge_lora` quando quiseres modelo fundido)
- `**data/**` — `raw/`, `snapshots/`, exemplos.
- `**prompt_system/**` — ficheiros de prompt usados no fluxo de dados, se aplicável.
- `**treina.sh**`, `**prompt.sh**` — atalhos (a partir da raiz do repositório).

Documentação longa: `[../README.md](../README.md)` (guia geral) e ficheiros `README_DATASETS.md`, `README_SNAPSHOT.md`, `README_INFERIR.md` **nesta pasta**.

Saídas de treino (adapter, merge): por defeito `**trein/outputs/lora_adapter`** e `**trein/outputs/merged_model**` (nomes ajustáveis com `--output_dir` / `merge`).

**Treino real** = `python3 trein/train_lora.py …` a partir da **raiz** do repositório (o script assume imports e caminhos relativos a isso). Atalho: `./trein/treina.sh` cria/usa o venv e, no fim, chama o `train_lora` com o comando que estiver no **final** de `treina.sh` (hoje: `--train_file` de exemplo; **substitui** essa linha para o teu dataset e flags).
