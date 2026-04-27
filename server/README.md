# Servidor local LoRA (`serve_lora.py`)

Este diretório contém o **servidor HTTP** que carrega o modelo **uma única vez** e responde a pedidos de chat. Assim não precisas de voltar a carregar vários gigabytes sempre que quiseres uma resposta (como aconteceria ao correr `trein/inferir.py` de cada vez).

## Variáveis de ambiente (PostgreSQL, etc.)

Copia `server/.env.example` para **`server/.env`** (ou cria `.env` na **raiz** do repo) e preenche pelo menos **`ORACULO_PG_HOST`** e o resto de `ORACULO_PG_*`. O `serve_lora.py` **não arranca** se não existir nenhum desses ficheiros. Carrega primeiro `.env` na raiz e depois **`server/.env`**, que **substitui** chaves repetidas.

## O que precisas antes

1. **Raiz do projeto** — a pasta *pai* desta (`server/`), com `server/requirements.txt` (e opcionalmente `requirements.txt` na raiz que junta treino+servidor), `trein/data_config.py` (config partilhada), `server/lora_engine.py` e `model_service/` (inferência + API `/v1` OpenAI-compat).

2. **Treino concluído** — normalmente existirá `trein/outputs/lora_adapter/` com o adapter gerado pelo `trein/train_lora.py` (ou um modelo fundido em `trein/outputs/merged_model/` após o `trein/merge_lora.py`).

3. **Dependências** — na raiz do projeto:

   ```bash
   pip install -r server/requirements.txt
   ```

   Inclui `fastapi` e `uvicorn` necessários para o servidor.

4. **Ambiente virtual (opcional)** — se usas `.venv` na raiz, o script `serve.sh` ativa-o automaticamente.

## Como iniciar o servidor

**Sempre a partir da raiz do projeto** (para caminhos e imports estarem corretos):

```bash
cd /caminho/para/TREINAMENTO\ LLM
python3 server/serve_lora.py
python3 server/serve_lora.py --base-only --model_name Qwen/Qwen3-8B --trust_remote_code
```

Atalho com venv:

```bash
./server/serve.sh
```

Na primeira execução vês na consola mensagens do tipo “A carregar modelo…”. Quando aparecer “Pronto”, o servidor escuta por defeito em **`0.0.0.0`** (todas as interfaces):

- **Nesta máquina:** `http://127.0.0.1:8765/`
- **Noutro PC ou telemóvel na mesma rede / VM na nuvem:** `http://IP-DO-SERVIDOR:8765/` (substitui pelo IP público ou privado do host)

**Firewall / nuvem:** abre a porta escolhida (ex. **8765/tcp**) no *security group*, *ufw* ou painel do teu fornecedor, senão o browser noutra máquina não liga mesmo com `0.0.0.0`.

**Só localhost** (não expor na rede): `python3 server/serve_lora.py --host 127.0.0.1`

## Interface no navegador

A página em `server/static/index.html` oferece um chat simples: escreves a mensagem, pressionas Enter ou “Enviar”, e a resposta aparece abaixo. O histórico da conversa é enviado de volta ao servidor para o modelo manter o contexto (dentro do limite do modelo).

## De onde vêm o modelo base e o LoRA

Os valores **por omissão** vêm de `../trein/data_config.py`:

| Definição | Significado |
|-----------|-------------|
| `DEFAULT_MODEL_NAME` | ID do modelo base no Hugging Face (tem de ser o **mesmo** usado no treino). |
| `DEFAULT_ADAPTER_DIR` | Pasta do adapter (por defeito `trein/outputs/lora_adapter`). |
| `DEFAULT_MERGED_MODEL_DIR` | Pasta do modelo fundido (`trein/outputs/merged_model`). |

Se alterares estes campos **uma vez** no `trein/data_config.py`, o servidor, o `trein/inferir.py` e o treino continuam alinhados.

## Modelo fundido vs base + LoRA

Ao arrancar, o script tenta usar o modelo **fundido** se existir `trein/outputs/merged_model/config.json`. Nesse caso carrega só essa pasta (mais simples para inferência).

Se não houver modelo fundido, carrega o **modelo base** (`DEFAULT_MODEL_NAME`) mais o adapter em `DEFAULT_ADAPTER_DIR`.

Podes forçar caminhos na linha de comando (ver abaixo).

## Inferência HF (PyTorch) vs GGUF (`.gguf` quantizado)

Podes levantar o **mesmo** servidor com o motor **HuggingFace** (padrão) ou com ficheiro **GGUF** via `llama-cpp-python` — útil em CPU / menos RAM. **Guia completo (env, dependências, exemplos):** [README_INFERENCE_HF_GGUF.md](README_INFERENCE_HF_GGUF.md).

- **Resumo:** `ORACULO_INFERENCE_BACKEND=gguf` + `ORACULO_GGUF_PATH` (ou ficheiro em `tools/quantized_model/…` se existir) e `pip install -r server/requirements-gguf.txt`.

## Definições de administrador: llama-server (HTTP)

Quando o backend é **llama-server**, o processo Python **não** carrega pesos HF nem GGUF: só faz de *proxy* para o binário `llama-server` (llama.cpp) que já tens a correr com `-m …`. Estas opções aparecem na UI (Definições) para **administradores** e gravam-se na base (tabela `app_global`). **Importante:** o URL efectivo no arranque do `serve_lora` (usar base de dados vs `.env`) só é aplicado depois de **reiniciares** o servidor Oráculo.

| Campo                           | Explicação prática                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usar llama-server**           | Define quem é o “cérebro”. **Ligado:** o Oráculo só faz requisições HTTP e quem processa tudo é o `llama-server`. **Desligado:** o Python tenta lidar diretamente (via `.env`). |
| **API — host / IP**             | Endereço onde o `llama-server` está rodando. Ex.: `127.0.0.1` (mesma máquina) ou IP remoto. É para onde o Oráculo envia as requisições.                                         |
| **Porta**                       | Porta HTTP do `llama-server` (ex.: `8080`). Forma a URL completa `http://host:porta`.                                                                                           |
| **Contexto n_ctx (referência)** (ex.: 512 – 32768+) | **Não afeta execução.** É só informativo. O valor real vem do `-c` no arranque do `llama-server`. Serve como lembrete do limite de contexto.                                    |
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
python3 server/serve_lora.py [--host 0.0.0.0] [--port 8765] \
  [--model_name ID_NO_HUB] \
  [--adapter_dir /caminho/para/lora_adapter] \
  [--merged_model_dir /caminho/para/merged_model] \
  [--inference-backend hf|gguf] [--gguf-path /caminho/modelo.gguf] \
  [--trust_remote_code]
```

**Padrão:** `--host 0.0.0.0` (acesso a partir de outras máquinas na rede, se o firewall permitir).

Exemplos:

- **Só este PC:** `--host 127.0.0.1`
- Outra porta: `--port 8080`
- Modelo base diferente do `data_config.py`: `--model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## API HTTP (para integrações)

Com o servidor em execução:

- **Estado** — `GET http://127.0.0.1:8765/api/status`  
  Resposta JSON: `loaded`, `mode` (ex. `fundido`, `base+LoRA` ou `gguf`), `backend` (`hf` ou `gguf`), `model_name`.

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
  - Se definires **`ORACULO_OPENAI_API_KEY`** em `server/.env`, todos os pedidos a `/v1/*` devem enviar `Authorization: Bearer <essa chave>`. Sem a variável, `/v1/*` aceita pedidos **sem** Bearer (só em rede confiável).  
  - O motor partilhado com estas rotas está no pacote **`model_service/`** (raiz do repo); este diretório `server/` continua a ser a camada web (FastAPI, auth, estáticos).

## Memória e desempenho

O processo do servidor **mantém o modelo na RAM** (ou na GPU, se tiveres CUDA) **enquanto estiver a correr**. O ganho é **não recarregar** pesos a cada pergunta. Fecha o servidor (Ctrl+C na consola) quando não precisares dele para libertar memória.

Só é processado **um pedido de geração de cada vez** (fila interna), para evitar picos de memória com dois chats em paralelo no mesmo processo.

## Resolução rápida de problemas

- **Erro ao importar módulos** — corre a partir da **raiz** do repo, não dentro de `server/` com outro `cwd`, a menos que ajustes o `PYTHONPATH` manualmente.
- **`ModuleNotFoundError: fastapi`** — `pip install -r server/requirements.txt` (a partir da raiz do repositório).
- **Modelo não carrega / pasta não encontrada** — confirma que `trein/outputs/lora_adapter` existe após o treino, ou passa `--adapter_dir` / treina de novo. Para fundido, verifica `config.json` dentro da pasta merged.
- **Porta ocupada** — usa `--port 8766` (ou outra porta livre).
