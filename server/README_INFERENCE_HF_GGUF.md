# Interruptor: inferência **HF (PyTorch)** vs **GGUF (llama-cpp-python)**

O mesmo servidor (`server/serve_lora.py`) pode arrancar de duas formas:

| Modo     | O que carrega | Dependências |
|----------|---------------|--------------|
| **hf**   | `transformers` + `peft` — merge fundido *ou* base + LoRA (pastas `trein/outputs/...`) | `server/requirements.txt` (torch, etc.) |
| **gguf** | Ficheiro **.gguf** (ex. Q4_K_M produzido com llama.cpp fora do Python)        | + `server/requirements-gguf.txt`         |

- **Não** é preciso trocar de projecto: a UI, a base de dados, auth e a API **HTTP** mantêm-se; muda **só** o motor de geração de texto.
- O modo **gguf** **não** lê o `merged_model` em safetensors: usa **apenas** o caminho para o ficheiro `.gguf` (já alinhado ao *chat template* do modelo na conversão).

## 1) Instalação (GGUF)

Na raiz do repositório:

```bash
pip install -r server/requirements.txt
pip install -r server/requirements-gguf.txt
```

Se `import llama_cpp` falhar, veja a [documentação de instalação do llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/installation/) (CPU vs GPU/CUDA, `CMAKE_ARGS`, *wheels* oficiais).

## 2) Onde fica o ficheiro .gguf

- Caminho padrão sugerido no repo (se o ficheiro existir):  
  `tools/quantized_model/Merged_Model-3.1B-Q4_K_M.gguf`  
  (definido em `trein/data_config.py` como `DEFAULT_GGUF_PATH`.)
- Pode apontar para outro sítio com **variável de ambiente** ou **CLI** (ver abaixo).

## 3) Arranque

### Só variáveis de ambiente (ficheiro `server/.env` ou a shell)

- Modo **GGUF**:
  - `ORACULO_INFERENCE_BACKEND=gguf`
  - `ORACULO_GGUF_PATH=/caminho/absoluto/para/meu_modelo_Q4_K_M.gguf` (se não usar o ficheiro default)
- Modo **HF** (comportamento anterior):
  - `ORACULO_INFERENCE_BACKEND=hf` (ou omitir, é o padrão)
  - `ORACULO_MERGED_MODEL_DIR` / `ORACULO_ADAPTER_DIR` etc., como antes

### Linha de comando (sobrepõe o default do env, quando indicado)

```bash
# GGUF: precisa de um ficheiro .gguf acessível
cd /caminho/raiz/do/treinamento_llms
python3 server/serve_lora.py --inference-backend gguf --gguf-path tools/quantized_model/Merged_Model-3.1B-Q4_K_M.gguf

# HF: igual ao que já usava
python3 server/serve_lora.py --inference-backend hf
# ou simplesmente (hf é o padrão)
python3 server/serve_lora.py
```

A partir da raiz do projecto, como sempre (imports e caminhos relativos).

## 4) Tuning (GGUF)

Opcionais (ambiente / `.env`):

| Variável | Efeito (resumo) |
|----------|-----------------|
| `ORACULO_GGUF_N_CTX` | Tamanho de contexto (default `4096`). |
| `ORACULO_GGUF_N_GPU_LAYERS` | Camadas na GPU: `-1` = todas; `0` = só CPU. |

## 5) API e estado

- `GET /api/status` passa a incluir `"backend": "hf"` ou `"backend": "gguf"` quando o modelo está carregado.
- `POST /v1/chat/completions` e o chat da UI usam o mesmo *runtime*; no modo GGUF, os *tokens* de *usage* vêm de `llama_cpp` quando disponível.

## 6) Obter o .gguf

A conversão HF → GGUF fica fora deste `serve_lora` (ex.: *scripts* do [llama.cpp](https://github.com/ggerganov/llama.cpp), `convert_hf_to_gguf.py`, depois *quantize*). O teu repositório `tools/quantized_model/` no servidor é um sítio natural para guardar o ficheiro após a conversão.

## 7) Só um modo por processo

Cada instância do Uvicorn carrega **um** *backend* no arranque. Para testar A/B (HF vs GGUF), levanta duas instâncias em **portas** diferentes, com `ORACULO_INFERENCE_BACKEND` (e caminhos) distintos, ou com um *reverse proxy* a encaminhar.
