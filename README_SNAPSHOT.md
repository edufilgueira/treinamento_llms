# Lógica do snapshot de treino

Este documento explica **só** o fluxo entre `data/raw/`, `build_snapshot.py`, `data/snapshots/` e `train_lora.py`. Para estrutura de pastas e curadoria de dados, veja [README_DATASETS.md](README_DATASETS.md). Para a pipeline completa de LoRA, veja [README.md](README.md).

---

## 1. Objetivo

- Você mantém **vários** arquivos `.jsonl` em `data/raw/` (por tema ou fonte).
- **`build_snapshot.py`** gera **um único** arquivo em `data/snapshots/` no formato que o treino espera (apenas `messages` por linha).
- **`train_lora.py`** lê esse snapshot **sem** você precisar passar o caminho toda vez — desde que não use `--train_file`.

Nenhum script importa o outro: você roda `build_snapshot.py` **manualmente** quando quiser atualizar o snapshot; o treino **só lê** arquivos já gerados.

---

## 2. Arquivos envolvidos

| Arquivo | Papel |
|---------|--------|
| **`data_config.py`** | Define `DATASET_VERSION`, `SNAPSHOT_DATE_PREFIX`, pastas `RAW_DIR` e `SNAPSHOTS_DIR`, e as funções `snapshot_output_path()`, `find_best_snapshot()`, `resolve_train_file()`. |
| **`build_snapshot.py`** | Percorre **recursivamente** todos os `.jsonl` em `data/raw/`, valida `messages`, opcionalmente embaralha, grava em `data/snapshots/train_<prefixo>_<versão>.jsonl`. |
| **`train_lora.py`** | Importa `resolve_train_file` de `data_config.py`. Se `--train_file` não for passado, usa o snapshot escolhido por `find_best_snapshot()`. |

---

## 3. Nome do snapshot e sobrescrita

O caminho de saída do `build_snapshot.py` é sempre:

```text
data/snapshots/train_<SNAPSHOT_DATE_PREFIX>_<DATASET_VERSION>.jsonl
```

Exemplo com os valores padrão iniciais:

- `DATASET_VERSION = "v1"`
- `SNAPSHOT_DATE_PREFIX = "2025-03"`
- → **`data/snapshots/train_2025-03_v1.jsonl`**

**Enquanto você não mudar** `DATASET_VERSION` nem `SNAPSHOT_DATE_PREFIX` em `data_config.py`, cada novo `python build_snapshot.py` **sobrescreve** esse mesmo arquivo. Assim você evita acumular dezenas de arquivos idênticos em versão “v1”.

Quando quiser uma **linha nova** de snapshot (por exemplo, congelar v1 e passar a trabalhar em v2):

1. Edite `data_config.py`: por exemplo `DATASET_VERSION = "v2"` e `SNAPSHOT_DATE_PREFIX = "2025-04"`.
2. Rode `python build_snapshot.py` de novo.
3. Será criado **`train_2025-04_v2.jsonl`**; o arquivo `train_2025-03_v1.jsonl` **permanece** no disco (o build não apaga versões antigas).

---

## 4. Como o `train_lora.py` escolhe qual snapshot usar

Isso vale **apenas** quando você **não** passa `--train_file`.

1. `resolve_train_file(None)` chama `find_best_snapshot()`.
2. `find_best_snapshot()` lista todos os arquivos em `data/snapshots/` que casam com o padrão glob `train_*_v*.jsonl`.
3. Para cada arquivo, extrai o número da versão do sufixo `_v<number>.jsonl` (ex.: `v2` → 2).
4. Escolhe o arquivo com **maior número de versão** (ex.: **v2 vence v1**).
5. Se dois arquivos tiverem a **mesma** versão (situação rara), ganha o de **maior tamanho em bytes**.

Exemplo: existem `train_2025-03_v1.jsonl` e `train_2025-04_v2.jsonl` → o treino usa o **v2**.

6. Se **não** existir nenhum snapshot válido, o próximo fallback é `data/exemplo_treino.jsonl` (se existir).
7. Se não houver snapshot nem exemplo, o programa encerra com erro pedindo para rodar `build_snapshot.py` (ou passar `--train_file`).

### Forçar um arquivo específico

Sempre que você passar:

```bash
python train_lora.py --train_file caminho/qualquer.jsonl
```

a lógica de “maior versão” **não** é usada para esse argumento; o treino lê **exatamente** esse caminho.

---

## 5. O que o `build_snapshot.py` faz nos dados

- **Entrada:** todos os `.jsonl` sob `data/raw/` (subpastas incluídas).
- **Linha válida:** JSON com chave **`messages`** (lista de `role` / `content`). Outras chaves (ex.: `meta`) são **descartadas** na saída — só vai `{"messages": ...}`.
- **Linhas inválidas** ou sem `messages`: são ignoradas com aviso no stderr.
- **Ordem:** por defeito as linhas válidas são **embaralhadas** (semente fixa `--seed`, por padrão 42). Use `--no-shuffle` para manter ordem determinística (concatenação ordenada por caminho de arquivo).

---

## 6. Fluxo típico de trabalho

1. Coloque ou atualize `.jsonl` em `data/raw/...`.
2. Ajuste `data_config.py` se estiver subindo de versão ou mudando o prefixo.
3. Rode **`python build_snapshot.py`**.
4. Rode **`python train_lora.py`** (e demais flags de treino). Sem `--train_file`, usa o snapshot de **maior versão**.

---

## 7. Resumo em uma frase

**Snapshot** = uma “foto” concatenada (e opcionalmente embaralhada) de tudo que está em `raw/`, com nome fixo por `(prefixo, versão)`; o **treino automático** sempre prefere o arquivo `train_*_v*.jsonl` com **maior `v`**, salvo se você passar `--train_file`.
