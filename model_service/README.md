# model_service

Pacote com o **runtime de inferência** (carregar modelo uma vez, gerar texto, fila global) e as rotas HTTP **compatíveis com a API OpenAI** (`/v1/...`). O servidor web em `server/serve_lora.py` importa este pacote, carrega o modelo no arranque e **monta o mesmo `FastAPI`** — não é um processo separado por defeito.

## O que fica aqui

| Ficheiro | Função |
|----------|--------|
| `runtime.py` | `ModelRuntime` + `get_runtime()`: `load()` / `load_gguf()`, `generate()`, `stream()`, lock de um pedido de geração de cada vez. |
| `gguf_engine.py` | Carga e *chat* com ficheiro `.gguf` via `llama-cpp-python` (quando `ORACULO_INFERENCE_BACKEND=gguf`). |
| `openai_routes.py` | Router FastAPI com prefixo `/v1`: chat completions e listagem de modelos. |

A implementação de baixo nível com **HuggingFace** (streaming, LoRA) continua em `server/lora_engine.py`. O modo **GGUF** não usa esse ficheiro; ver [server/README_INFERENCE_HF_GGUF.md](../server/README_INFERENCE_HF_GGUF.md).

## Quem precisa desta API

- **Interface web do projeto** (`server/static/…`) usa sobretudo `/api/chat`, `/api/chat/jobs`, etc., com **sessão** no browser. **Não** é obrigatório instalar o SDK `openai` para usar o site.
- **`/v1/*`** destina-se a **outras aplicações** que já falam o dialecto OpenAI: scripts Python, n8n, ferramentas com `base_url` customizado, ou qualquer cliente que envie `POST /v1/chat/completions` com JSON no estilo OpenAI.

## Autenticação

Variável de ambiente (ficheiro `server/.env` ou raiz `.env`, conforme o `serve_lora` carrega):

| `ORACULO_OPENAI_API_KEY` | Comportamento |
|--------------------------|----------------|
| **Definida** (não vazia) | Todos os pedidos a `/v1/*` devem incluir `Authorization: Bearer <valor exato da variável>`. Caso contrário: **401**. |
| **Não definida** ou vazia | Pedidos a `/v1/*` **sem** Bearer são aceites. Use só em ambiente confiável (ex.: desenvolvimento local). |

Isto **não** substitui o login da interface web; é independente.

## Endpoints

Base: `http://<host>:<porta>` (por defeito porta **8765**). Os paths abaixo são relativos a essa base.

### `GET /v1/models`

- **Resposta:** objeto com `object: "list"` e `data`: lista de modelos.
- Se o modelo **ainda não estiver carregado** (ou arranque em `--ui-only`), `data` vem **vazia** `[]`.
- Se estiver carregado, há uma entrada com `id` igual ao identificador Hugging Face usado no arranque (`model_name` / `data_config`).

### `POST /v1/chat/completions`

Corpo JSON (campos extra ignorados, como na API OpenAI):

| Campo | Tipo | Notas |
|-------|------|--------|
| `model` | string | Opcional; eco na resposta. O modelo real é o já carregado no servidor. |
| `messages` | array | Obrigatório. Cada item: `role` (`system`, `user`, `assistant`) e `content` (string). |
| `max_tokens` | int | Por defeito 2048; máximo 4096. |
| `temperature` | float | Por defeito 0.7. |
| `top_p` | float | Por defeito 0.9. |
| `stream` | bool | `false`: JSON completo; `true`: `text/event-stream` (SSE) estilo `chat.completion.chunk`. |

**Não suportado:** mensagens com `content: null`, `tool` / function calling, imagens, etc. — pedidos assim podem devolver **400**.

**Resposta (sem stream):** objeto com `id`, `object: "chat.completion"`, `created`, `model`, `choices[]` (com `message.role` / `message.content`), `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).

**Resposta (stream):** linhas SSE `data: { ... }` e termina com `data: [DONE]`.

**Erros comuns:** **503** se o servidor estiver em modo só UI ou o modelo não tiver carregado; **401** se a chave Bearer estiver errada ou em falta quando `ORACULO_OPENAI_API_KEY` está definida.

## Exemplos

### curl (com chave configurada no servidor)

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer uma_chave_segura" \
  -d '{"model":"local","messages":[{"role":"user","content":"Olá"}],"max_tokens":256}'
```

### Cliente Python oficial OpenAI (outra aplicação, não o servidor)

Instalação: `pip install openai` (no ambiente **do cliente**, não é requisito do `server/requirements.txt` para o site).

A string de `api_key` deve ser **igual** à `ORACULO_OPENAI_API_KEY` quando essa variável está definida no servidor.

```python
from openai import OpenAI

client = OpenAI(
    api_key="uma_chave_segura",
    base_url="http://127.0.0.1:8765/v1",
)

r = client.chat.completions.create(
    model="qualquer-nome-eco",
    messages=[{"role": "user", "content": "Olá"}],
)
print(r.choices[0].message.content)
```

## Resumo

- **`model_service`** = lógica partilhada de inferência + contrato **OpenAI-like** em `/v1`.
- **Site Oráculo** continua a usar **`/api/...`** com sessão; **não** precisas do SDK OpenAI para o dia a dia no browser.
- Usa **`/v1`** quando quiseres integrar **programas externos** com o mesmo formato que a API da OpenAI.
