"""
Configuração compartilhada: dados (snapshots) e defaults do modelo / pastas de saída.

**Modelo e pastas** — `DEFAULT_MODEL_NAME`, `DEFAULT_ADAPTER_DIR` e `DEFAULT_MERGED_MODEL_DIR`
são usados por `train_lora.py`, `inferir.py`, `merge_lora.py` e `server/serve_lora.py`. Altere **uma vez** aqui para
manter treino, inferência e merge alinhados. Opcional: `DEFAULT_GGUF_PATH` (ficheiro .gguf, modo `ORACULO_INFERENCE_BACKEND=gguf`).

**Progresso em `inferir.py`** — duas coisas **distintas**:
- `SHOW_LOADING_PROGRESS` — **só** barras do tqdm (pesos) e do download no Hugging Face Hub (`True` = mostrar, `False` = ocultar).
- `FIX_GENERATION_LENGTH_CONFLICT` — ajuste do `generation_config` para evitar o aviso do Transformers sobre `max_new_tokens` vs `max_length` no `generate()`; **não** é barra de UI e **não** segue `SHOW_LOADING_PROGRESS`.

**Snapshots** — edite a versão e o prefixo de data:
- Enquanto DATASET_VERSION e SNAPSHOT_DATE_PREFIX não mudarem, o build_snapshot.py
  **sobrescreve** o mesmo arquivo (ex.: train_2025-03_v1.jsonl).
- Ao subir para v2, altere DATASET_VERSION para "v2" e SNAPSHOT_DATE_PREFIX para um
  novo período (ex.: "2025-04"); aí será criado train_2025-04_v2.jsonl e o treino
  passará a preferir a **maior versão** presente em trein/data/snapshots/ (ver resolve_train_file).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# --- Barras de progresso (tqdm / Hugging Face Hub) em inferir.py ---
# True  → mostra barras ao carregar pesos e ao baixar do Hub.
# False → suprime (TQDM_DISABLE e HF_HUB_DISABLE_PROGRESS_BARS).
SHOW_LOADING_PROGRESS = True


# --- Geração (inferir.py): aviso max_new_tokens vs max_length do checkpoint ---
# True  → define generation_config.max_length = None antes do generate (recomendado).
# False → não altera; pode voltar o aviso na consola ao gerar. Independente de SHOW_LOADING_PROGRESS.
FIX_GENERATION_LENGTH_CONFLICT = True


def apply_loading_progress_env() -> None:
    """Aplica SHOW_LOADING_PROGRESS ao processo atual (chamar antes de importar transformers/tqdm)."""
    if not SHOW_LOADING_PROGRESS:
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    else:
        os.environ.pop("TQDM_DISABLE", None)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


# Pasta `trein/` (datasets, snapshots, outputs de adapter/merge relativos a ela).
TREIN_ROOT = Path(__file__).resolve().parent
# Raiz do repositório (pai de `trein/` e `server/`).
REPO_ROOT = TREIN_ROOT.parent
ROOT = TREIN_ROOT

# --- Modelo base no Hugging Face (mesmo ID em treino, inferência e merge) ---
# Use modelos **causais de texto** (Chat/Instruct): TinyLlama, Llama, Qwen2, Mistral, etc.
# Não use modelos **multimodais** (ex.: Janus, LLaVA) — o train_lora usa AutoModelForCausalLM
# e o erro "JanusConfig ... AutoModelForCausalLM" indica modelo incompatível com esta pipeline.
#
# Repositórios **só GGUF** (ex.: *...-GGUF* da bartowski/TheBloke) são para llama.cpp/Ollama —
# não trazem tokenizer + pesos PyTorch como o `transformers` espera. Para este pipeline use um
# modelo **HF completo** (safetensors); depois pode converter para GGUF à parte.
#
# `meta-llama/Llama-3.2-3B-Instruct` é **gated**: pede acesso no Hub + `huggingface-cli login`.
# Para Llama 3.2 uncensored em HF (aberto): `chuanli11/Llama-3.2-3B-Instruct-uncensored`
#
# Qwen 3B: em GPU usa FP16/BF16; em CPU (FP32) treino LoRA é possível mas pode exigir RAM
# confortável — train_lora avisa neste caso; TinyLlama é alternativa para CPUs mais limitadas.
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"

# --- Pastas de saída do LoRA / merge: sempre trein/outputs/... na raiz do repositório ---
DEFAULT_ADAPTER_DIR = REPO_ROOT / "trein" / "outputs" / "lora_adapter_qwen3-8b"
DEFAULT_MERGED_MODEL_DIR = REPO_ROOT / "trein" / "outputs" / "merged_qwen3-8b"
# Ficheiro GGUF sugerido (llama.cpp / `llama-cpp-python`); ajuste ou use ORACULO_GGUF_PATH.
DEFAULT_GGUF_PATH = REPO_ROOT / "tools" / "quantized_model" / "Merged_Model-3.1B-Q4_K_M.gguf"

# Versão do snapshot: "v1", "v2", ... (aparece no nome do ficheiro)
DATASET_VERSION = "v1"

# Prefixo de data/período no nome do ficheiro (ex.: 2025-03 → train_2025-03_v1.jsonl)
SNAPSHOT_DATE_PREFIX = "2025-03"

SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
RAW_DIR = ROOT / "data" / "raw"


def snapshot_output_path() -> Path:
    """Ficheiro que build_snapshot.py grava (sobrescreve se PREFIX+VERSION forem iguais)."""
    return SNAPSHOTS_DIR / f"train_{SNAPSHOT_DATE_PREFIX}_{DATASET_VERSION}.jsonl"


def _snapshot_version_key(path: Path) -> tuple[int, int]:
    """Ordenação: maior número de versão (v2 > v1); empate → maior tamanho em bytes."""
    m = re.search(r"_v(\d+)\.jsonl$", path.name, re.IGNORECASE)
    if not m:
        return (-1, -1)
    return (int(m.group(1)), path.stat().st_size)


def find_best_snapshot(snapshots_dir: Path | None = None) -> Path | None:
    """
    Entre todos os ficheiros train_*_v*.jsonl em snapshots/, escolhe o de **maior versão**
    (v3 > v2 > v1). Se duas versões forem iguais, prefere o ficheiro **maior** em bytes.
    """
    base = snapshots_dir or SNAPSHOTS_DIR
    if not base.is_dir():
        return None
    candidates = list(base.glob("train_*_v*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=_snapshot_version_key)


EXAMPLE_FALLBACK = ROOT / "data" / "raw" / "exemplo" / "exemplo_treino.jsonl"


def resolve_train_file(explicit: Path | None) -> Path:
    """
    Se `explicit` for passado, usa esse caminho.
    Caso contrário: maior versão em SNAPSHOTS_DIR; se não houver snapshot,
    usa trein/data/raw/exemplo/exemplo_treino.jsonl se existir.
    """
    if explicit is not None:
        return explicit
    best = find_best_snapshot()
    if best is not None:
        return best
    if EXAMPLE_FALLBACK.is_file():
        return EXAMPLE_FALLBACK
    raise FileNotFoundError(
        "Nenhum snapshot em trein/data/snapshots/ (train_*_v*.jsonl) nem exemplo em "
        "trein/data/raw/exemplo/. Rode: python trein/build_snapshot.py (na raiz do repo)."
    )
