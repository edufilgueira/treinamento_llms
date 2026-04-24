# Inferência local com `inferir.py`

Este documento descreve **só** o script **`inferir.py`**: o que ele faz por dentro, argumentos e uso prático. O guia geral da pipeline (treino, merge, cache) está no [README.md](README.md).

---

## 1. Para que serve

- **Testar** no seu computador se o **adapter LoRA** treinado (`train_lora.py`) carrega e gera texto.
- Executa **uma geração por comando** e imprime o resultado no **terminal** — não é um servidor HTTP nem API de produção.

Se você **fundiu** o modelo (`merge_lora.py`), use `AutoModelForCausalLM.from_pretrained` na pasta fundida; o `inferir.py` atual está pensado para **base do Hub + pasta do adapter** (PEFT).

---

## 2. Lógica interna (ordem do que acontece)

1. **Tokenizer** — carregado do **`--model_name`** (mesmo ID do Hugging Face usado no treino). Se não houver `pad_token`, usa `eos` como pad.
2. **Modelo base** — `AutoModelForCausalLM.from_pretrained` com `dtype` adequado: fp32 em CPU; bf16 ou fp16 em GPU quando disponível.
3. **Adapter** — `PeftModel.from_pretrained(modelo_base, adapter_dir)` injeta os pesos LoRA salvos em `--adapter_dir`.
4. **Dispositivo** — em CPU, o modelo é enviado explicitamente para CPU; em GPU usa `device_map="auto"`.
5. **Prompt** — monta `messages = [{"role": "user", "content": --prompt}]` e aplica **`apply_chat_template`** com `add_generation_prompt=True` para obter tensores no formato que o modelo chat espera.
6. **Geração** — `model.generate(**model_inputs, ...)` com os parâmetros fixos descritos na §4.
7. **Saída** — `decode` do primeiro batch e `print` do texto completo (inclui prompt formatado conforme o template).

Tudo roda sob `torch.no_grad()` (inferência, sem treino).

---

## 3. Pré-requisitos

- **Python do projeto (venv):** use sempre o interpretador que tem as dependências instaladas. Se aparecer `ModuleNotFoundError: No module named 'peft'`, costuma ser porque rodou `python` do sistema em vez do venv.

  Na raiz do projeto:

  ```bash
  cd "/PROJETO/ORÁCULO SANTUÁRIO DIGITAL/TREINAMENTO LLM"
  source .venv/bin/activate
  pip install -r trein/requirements.txt
  python inferir.py --prompt "O que é um oráculo digital?"
  ```
  ou

  ```bash
  python3 -m venv .venv   # só se ainda não existir
  source .venv/bin/activate
  pip install -r trein/requirements.txt
  ```

  Sem ativar o venv, chame o script assim:

  ```bash
  .venv/bin/python inferir.py --prompt "O que é um oráculo digital?"
  ```

  Depois de `source .venv/bin/activate`, confira com `which python` — o caminho deve ser `.../TREINAMENTO LLM/.venv/bin/python`.

- Dependências (`peft`, `transformers`, `torch`, etc.) — ver também [README.md](README.md).
- Os valores padrão de **`--model_name`** e **`--adapter_dir`** vêm de **`data_config.py`** (`DEFAULT_MODEL_NAME`, `DEFAULT_ADAPTER_DIR`) — os mesmos usados em `train_lora.py` e `merge_lora.py`. Altere o modelo **uma vez** nesse ficheiro para alinhar toda a pipeline.
- Ter rodado **`train_lora.py`** antes e existir pasta em **`--adapter_dir`** com adapter + tokenizer (ou pelo menos `adapter_config.json` e pesos LoRA).
- **`--model_name` idêntico** ao usado no treino — senão arquitetura/template não batem com o adapter.

---

## 4. Argumentos de linha de comando

| Argumento | Padrão | Função |
|-----------|--------|--------|
| `--model_name` | `DEFAULT_MODEL_NAME` em `data_config.py` | ID do modelo **base** no Hub — **o mesmo** do `train_lora.py`. |
| `--adapter_dir` | `DEFAULT_ADAPTER_DIR` em `data_config.py` | Pasta onde o treino gravou o adapter (e tokenizer, se houver). |
| `--prompt` | texto de exemplo sobre oráculo digital | Texto do **usuário**; vira a mensagem `user` no chat template. |
| `--max_new_tokens` | `256` | Limite de **tokens novos** gerados após o prompt (não inclui o prompt). |
| `--trust_remote_code` | (desligado) | Ative se o cartão do modelo no Hub exigir código customizado. |

Ajuda: `python inferir.py --help`

---

## 5. Parâmetros de geração (fixos no código)

Não há flags na linha de comando para estes; para mudar, edite `inferir.py`:

| Parâmetro | Valor | Efeito |
|-----------|-------|--------|
| `do_sample` | `True` | Amostragem estocástica (não greedy). |
| `temperature` | `0.7` | Mais alto → mais variável; mais baixo → mais conservador. |
| `top_p` | `0.9` | Nucleus sampling (massa de probabilidade acumulada). |

---

## 6. Uso na prática

Com o venv ativado (`source .venv/bin/activate`) ou com `.venv/bin/python` no lugar de `python`.

**Teste básico** (adapter na pasta padrão):

```bash
python inferir.py --prompt "O que é um oráculo digital?"
```

**Outro adapter ou modelo base** (deve coincidir com o treino):

```bash
python inferir.py \
  --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --adapter_dir trein/outputs/lora_adapter \
  --prompt "Sua pergunta aqui" \
  --max_new_tokens 128
```

**Primeira execução:** o modelo base pode ser **baixado** para o cache do Hugging Face (`~/.cache/huggingface/hub/`); pode demorar.

**CPU:** geração é lenta em relação a GPU; é normal.

---

## 7. Limitações (esperado)

- **Carrega o modelo a cada execução** — para testes manuais está ok; para **produção** costuma-se usar um servidor que mantém o modelo em memória (API, vLLM, etc.).
- **Um prompt por execução** — sem fila, sem autenticação, sem streaming na CLI.
- Saída mistura **marcadores do template** (ex.: `<|user|>`, `<|assistant|>`) conforme o tokenizer do modelo — é o comportamento típico de `decode` completo.

---

## 8. Ligação com o resto do projeto

| Script | Relação |
|--------|---------|
| `train_lora.py` | Produz `--adapter_dir`. |
| `merge_lora.py` | Gera modelo **fundido**; o `inferir.py` **não** carrega essa pasta por padrão — usa base + adapter. |
| [README.md](README.md) | Instalação, treino, merge, cache. |
