# Oráculo — `server_for_serveless` (projeto à parte de `server/`)

Pacote Python **`server_for_serveless`** na raiz do repo. Arranque:

```bash
cd /caminho/para/treinamento_llms
python3 server_for_serveless/serve_lora.py
# ou: ./server_for_serveless/serve.sh
```

**Runpod:** `ORACULO_RUNPOD_ENDPOINT_ID` + `ORACULO_RUNPOD_API_KEY` em `server_for_serveless/.env` (ver `.env.example` nesta pasta). Arquitetura, fluxo da API e workers: **[README_RUNPOD_SERVERLESS.md](README_RUNPOD_SERVERLESS.md)**. Sem essas variáveis, usa-se llama-server HTTP como no projeto original.

---

Este diretório contém o **servidor HTTP** (FastAPI) que delega inferência ao **llama-server** ou ao **Runpod Serverless**; não carrega `.gguf` no processo Python.

## Variáveis de ambiente (PostgreSQL, etc.)

Copia `server_for_serveless/.env.example` para **`server_for_serveless/.env`** (ou cria `.env` na **raiz** do repo) e preenche pelo menos **`ORACULO_PG_HOST`**. Carrega primeiro `.env` na raiz e depois **`server_for_serveless/.env`**, que **substitui** chaves repetidas.

## Estrutura do código (`server_for_serveless/`)

| Pasta / ficheiro | Função |
|------------------|--------|
| `serve_lora.py` | Entrada: `PYTHONPATH`, `.env`, chama `server_for_serveless.main`. |
| `main.py` | `FastAPI`, *lifespan*, middleware, inclui routers. |
| `bootstrap.py` | Carrega `.env` e valida `ORACULO_PG_HOST`. |
| `api/v1/` | Rotas OpenAI-compat (`/v1/chat/completions`, `/v1/models`). |
| `api/web/` | Chat, auth, sessões, páginas estáticas. |
| `inference/` | *Runtime*, llama HTTP e cliente Runpod. |
| `db/` | PostgreSQL (`pg_db`, `auth_db`, `chat_sessions_db`). |
| `schemas/` | Modelos Pydantic da API web. |
| `services/` | Lógica sem HTTP (ex. preferências de chat). |
| `static/` | UI (HTML, JS, CSS). |

## O que precisas antes

1. **Raiz do projeto** — pasta que contém `server_for_serveless/`, com `server_for_serveless/requirements.txt`.

2. **Inferência** — ou **llama-server** noutro host (URL no `.env` / admin), ou **Runpod** (`ORACULO_RUNPOD_*`). Se não definires URL e não usares Runpod, usa-se **`http://127.0.0.1:8080`** (exceto com `ORACULO_LLAMA_CPP_REQUIRE_EXPLICIT_URL=1`).

3. **Dependências** — na raiz do projeto:

   ```bash
   pip install -r server_for_serveless/requirements.txt
   ```

   Inclui `fastapi`, `uvicorn` e `httpx`; **não** inclui PyTorch nem `transformers`.

4. **Ambiente virtual (opcional)** — se usas `.venv` na raiz, o script `serve.sh` ativa-o automaticamente.

## Como iniciar o servidor

**Sempre a partir da raiz do projeto** (para caminhos e imports estarem corretos):

```bash
cd /caminho/para/TREINAMENTO\ LLM
python3 server_for_serveless/serve_lora.py
```

Atalho com venv:

```bash
./server_for_serveless/serve.sh
```

Na consola deves ver “Modo llama-server” e o URL upstream. Quando aparecer “Pronto”, o servidor escuta por defeito em **`0.0.0.0`** (todas as interfaces):

- **Nesta máquina:** `http://127.0.0.1:8765/`
- **Noutro PC ou telemóvel na mesma rede / VM na nuvem:** `http://IP-DO-SERVIDOR:8765/` (substitui pelo IP público ou privado do host)

**Firewall / nuvem:** abre a porta escolhida (ex. **8765/tcp**) no *security group*, *ufw* ou painel do teu fornecedor, senão o browser noutra máquina não liga mesmo com `0.0.0.0`.

**Só localhost** (não expor na rede): `python3 server_for_serveless/serve_lora.py --host 127.0.0.1`

## Interface no navegador

A página em `server_for_serveless/static/index.html` oferece um chat simples: escreves a mensagem, pressionas Enter ou “Enviar”, e a resposta aparece abaixo. O histórico da conversa é enviado de volta ao servidor para o modelo manter o contexto (dentro do limite do modelo).

## Definições de administrador: llama-server (HTTP)

O processo Python **só** faz de *proxy* para o `llama-server` (llama.cpp) com `-m …`. Estas opções aparecem na UI (Definições) para **administradores** e gravam-se na base (tabela `app_global`). **Importante:** o URL efectivo no arranque (base de dados vs `.env`) só é aplicado depois de **reiniciares** o servidor Oráculo. Se **Usar llama-server** estiver desligado no admin, é obrigatório **`ORACULO_LLAMA_CPP_BASE_URL`** no `.env`.

| Campo                           | Explicação prática                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usar llama-server**           | **Ligado:** URL do upstream vem do host/porta guardados na base (substitui `ORACULO_LLAMA_CPP_BASE_URL` no arranque). **Desligado:** usa só `ORACULO_LLAMA_CPP_BASE_URL` no `.env` (obrigatório). |
| **API — host / IP**             | Endereço onde o `llama-server` está rodando. Ex.: `127.0.0.1` (mesma máquina) ou IP remoto. É para onde o Oráculo envia as requisições.                                         |
| **Porta**                       | Porta HTTP do `llama-server` (ex.: `8080`). Forma a URL completa `http://host:porta`.                                                                                           |
| **Contexto n_ctx** | **Oráculo** trunca mensagens (da conversa mais antiga para a mais recente) para caber neste orçamento, deixando espaço para «Máx. tokens novos». O **llama-server** deve ter `-c` **≥** este valor.                                   |
| **Máx. tokens novos**  (ex.: 1 – 4096+)    | Limite de tokens da **resposta gerada** (`max_tokens`). Não inclui o prompt, só o que o modelo vai escrever.                                                                    |
| **Temperature**   (0.0 – 2.0)              | Controla aleatoriedade. Baixo = respostas mais diretas e previsíveis. Alto = mais criativas e variáveis.                                                                        |
| **Top P**  (0.0 – 1.0)                     | Filtro de probabilidade (*nucleus sampling*). Limita as escolhas aos tokens mais prováveis, evitando respostas muito aleatórias.                                                |
| **Repeat penalty**   (1.0 – 2.0)           | Evita repetição de palavras/frases. Valores típicos: `1.1 – 1.2`. Muito alto pode prejudicar coerência.                                                                         |
| **Repeat last N**  (0 – 4096+)             | Quantidade de tokens recentes considerados para aplicar a penalização de repetição. Ex.: `512` = olha os últimos 512 tokens.                                                    |
| **Reasoning**   (off / on / auto)          | Controla o modo “pensador” (em modelos que suportam). `off` = direto, `on` = raciocina antes, `auto` = comportamento padrão do modelo.                                      |
| **Reasoning budget**  (-1 – 4096+)         | Limite de “pensamento interno”. `0` ou `-1` geralmente desativa. Valores maiores permitem mais raciocínio, mas aumentam latência.                                               |


O **modelo** em si continua a ser o ficheiro `.gguf` que passas ao `llama-server` com `-m`. O Oráculo resolve o campo `model` da API com `ORACULO_LLAMA_CPP_MODEL` (opcional no `.env`) ou, se vazio, com o **primeiro** `id` devolvido por `GET /v1/models` no **arranque** do `serve_lora`.

## Opções da linha de comando

```text
python3 server_for_serveless/serve_lora.py [--host 0.0.0.0] [--port 8765] [--ui-only]
```

**Padrão:** `--host 0.0.0.0` (acesso a partir de outras máquinas na rede, se o firewall permitir).

Exemplos:

- **Só este PC:** `--host 127.0.0.1`
- Outra porta: `--port 8080`
## API HTTP (para integrações)

Com o servidor em execução:

- **Estado** — `GET http://127.0.0.1:8765/api/status`  
  Resposta JSON: `loaded`, `mode` (`llama_server`), `backend` (`llama_server`), `model_name`, `llama_server_url` quando aplicável.

- **Chat** — `POST http://127.0.0.1:8765/api/chat`  
  Corpo JSON (exemplo):

  ```json
  {
    "messages": [
      {"role": "user", "content": "Olá, quem és?"}
    ],
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }
  ```

  Podes incluir mensagens anteriores com `role` `user`, `assistant` ou `system`.

  Resposta:

  ```json
  {
    "reply": "texto gerado pelo assistente"
  }
  ```

Exemplo com `curl`:

```bash
curl -s http://127.0.0.1:8765/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá"}],"max_new_tokens":256}'
```

- **Chat em streaming (SSE)** — `POST http://127.0.0.1:8765/api/chat/stream` com o **mesmo JSON** que `/api/chat`. A resposta é `text/event-stream`: linhas `data: {"delta":"..."}` com pedaços do texto à medida que o modelo gera, e por fim `data: [DONE]`. A página `static/index.html` usa este endpoint para ir mostrando a resposta em tempo real.

- **OpenAI-compat** — `POST /v1/chat/completions`, `GET /v1/models` (subconjunto útil para clientes com `base_url` apontando para `http://<host>:<porta>/v1`). Corpo e resposta seguem o formato habitual (`messages`, `max_tokens`, `stream`, `choices`, `usage`, etc.). Ferramentas / `tool` roles não estão implementadas.  
  - Se definires **`ORACULO_OPENAI_API_KEY`** em `server_for_serveless/.env`, todos os pedidos a `/v1/*` devem enviar `Authorization: Bearer <essa chave>`. Sem a variável, `/v1/*` aceita pedidos **sem** Bearer (só em rede confiável).  
  - O *runtime* partilhado está em **`server_for_serveless/inference/`**; as rotas `/v1` em **`server_for_serveless/api/v1/`**.

## Memória e desempenho

O processo Oráculo **não** carrega o `.gguf`: a RAM/GPU do modelo fica no **llama-server**. Este processo mantém só ligações HTTP e estado da aplicação.

Só é processado **um pedido de geração de cada vez** no proxy (fila interna), para evitar sobrecarregar o upstream com dois chats em paralelo no mesmo worker.

## Resolução rápida de problemas

- **Erro ao importar módulos** — corre a partir da **raiz** do repo, não dentro de `server_for_serveless/` com outro `cwd`
- **`ModuleNotFoundError: fastapi`** — `pip install -r server_for_serveless/requirements.txt`
- **Modelo não carrega / 503** — confirma que o `llama-server` está a correr, que `ORACULO_LLAMA_CPP_BASE_URL` (ou admin com **Usar llama-server**) aponta para o host/porta certos, e reinicia o Oráculo após mudar definições.
- **Porta ocupada** — usa `--port 8766` (ou outra porta livre).


## Subir UI no Windows
```bash
python -m venv .venv e o Activate.ps1
cd C:\Users\Eduardo Filgueira\Documents\kiaia\treinamento_llms
.\.venv\Scripts\Activate.ps1
pip install -r equirements.txt
python serve_lora.py --ui-only
```