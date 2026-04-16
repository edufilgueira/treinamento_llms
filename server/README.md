# Servidor local LoRA (`serve_lora.py`)

Este diretório contém o **servidor HTTP** que carrega o modelo **uma única vez** e responde a pedidos de chat. Assim não precisas de voltar a carregar vários gigabytes sempre que quiseres uma resposta (como aconteceria ao correr `inferir.py` de cada vez).

## O que precisas antes

1. **Raiz do projeto** — a pasta *pai* desta (`server/`), onde estão `train_lora.py`, `data_config.py`, `lora_engine.py` e `requirements.txt`.

2. **Treino concluído** — normalmente existirá `outputs/lora_adapter/` com o adapter gerado pelo `train_lora.py` (ou um modelo fundido em `outputs/merged_model/` após o `merge_lora.py`).

3. **Dependências** — na raiz do projeto:

   ```bash
   pip install -r requirements.txt
   ```

   Inclui `fastapi` e `uvicorn` necessários para o servidor.

4. **Ambiente virtual (opcional)** — se usas `.venv` na raiz, o script `serve.sh` ativa-o automaticamente.

## Como iniciar o servidor

**Sempre a partir da raiz do projeto** (para caminhos e imports estarem corretos):

```bash
cd /caminho/para/TREINAMENTO\ LLM
python3 server/serve_lora.py
```

Atalho com venv:

```bash
./server/serve.sh
```

Na primeira execução vês na consola mensagens do tipo “A carregar modelo…”. Quando aparecer “Pronto”, abre o navegador em:

**http://127.0.0.1:8765/**

## Interface no navegador

A página em `server/static/index.html` oferece um chat simples: escreves a mensagem, pressionas Enter ou “Enviar”, e a resposta aparece abaixo. O histórico da conversa é enviado de volta ao servidor para o modelo manter o contexto (dentro do limite do modelo).

## De onde vêm o modelo base e o LoRA

Os valores **por omissão** vêm de `../data_config.py` (relativo a esta pasta — ou seja, `data_config.py` na raiz do projeto):

| Definição | Significado |
|-----------|-------------|
| `DEFAULT_MODEL_NAME` | ID do modelo base no Hugging Face (tem de ser o **mesmo** usado no treino). |
| `DEFAULT_ADAPTER_DIR` | Pasta do adapter (por defeito `outputs/lora_adapter`). |
| `DEFAULT_MERGED_MODEL_DIR` | Pasta do modelo fundido (`outputs/merged_model`). |

Se alterares estes campos **uma vez** no `data_config.py`, o servidor, o `inferir.py` e o treino continuam alinhados.

## Modelo fundido vs base + LoRA

Ao arrancar, o script tenta usar o modelo **fundido** se existir `outputs/merged_model/config.json`. Nesse caso carrega só essa pasta (mais simples para inferência).

Se não houver modelo fundido, carrega o **modelo base** (`DEFAULT_MODEL_NAME`) mais o adapter em `DEFAULT_ADAPTER_DIR`.

Podes forçar caminhos na linha de comando (ver abaixo).

## Opções da linha de comando

```text
python3 server/serve_lora.py [--host 127.0.0.1] [--port 8765] \
  [--model_name ID_NO_HUB] \
  [--adapter_dir /caminho/para/lora_adapter] \
  [--merged_model_dir /caminho/para/merged_model] \
  [--trust_remote_code]
```

Exemplos:

- Ouvir na rede local (outros dispositivos na LAN): `--host 0.0.0.0`
- Outra porta: `--port 8080`
- Modelo base diferente do `data_config.py`: `--model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## API HTTP (para integrações)

Com o servidor em execução:

- **Estado** — `GET http://127.0.0.1:8765/api/status`  
  Resposta JSON: `loaded`, `mode` (`fundido` ou `base+LoRA`), `model_name`.

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

## Memória e desempenho

O processo do servidor **mantém o modelo na RAM** (ou na GPU, se tiveres CUDA) **enquanto estiver a correr**. O ganho é **não recarregar** pesos a cada pergunta. Fecha o servidor (Ctrl+C na consola) quando não precisares dele para libertar memória.

Só é processado **um pedido de geração de cada vez** (fila interna), para evitar picos de memória com dois chats em paralelo no mesmo processo.

## Resolução rápida de problemas

- **Erro ao importar módulos** — corre a partir da **raiz** do repo, não dentro de `server/` com outro `cwd`, a menos que ajustes o `PYTHONPATH` manualmente.
- **`ModuleNotFoundError: fastapi`** — `pip install -r requirements.txt` na raiz.
- **Modelo não carrega / pasta não encontrada** — confirma que `outputs/lora_adapter` existe após o treino, ou passa `--adapter_dir` / treina de novo. Para fundido, verifica `config.json` dentro da pasta merged.
- **Porta ocupada** — usa `--port 8766` (ou outra porta livre).
