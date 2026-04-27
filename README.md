# Guia completo: pipeline de treino LoRA

Este documento explica **passo a passo** como usar os scripts, **para que serve** cada etapa e **o que cada variável** (argumento de linha de comando ou configuração interna) faz.

## Estrutura do repositório

- **`trein/`** — `data_config.py`, `train_lora.py`, `inferir.py`, `data/`, `outputs/`. [trein/README.md](trein/README.md).
- **`server/`** — API (FastAPI), `server/inference/`, `server/db/`, estáticos; **credenciais** em [server/.env.example](server/.env.example) → `server/.env`. Guia: [server/README.md](server/README.md).
- **Raiz** — `README.md` (este ficheiro); `requirements.txt` junta `trein/` + `server/`; ou instale só [trein/requirements.txt](trein/requirements.txt) / [server/requirements.txt](server/requirements.txt). (Opcional: `.env` na raiz; o Oráculo também lê `server/.env`, que tem prioridade.)

---

## Visão geral: o que esta pipeline faz

1. Você fornece um arquivo **JSONL** com conversas (`messages`: usuário + assistente).
2. O script **baixa** um modelo base do Hugging Face (por padrão o **TinyLlama** em formato chat). Para **pesquisar outros modelos** (filtros por tamanho, licença, etc.), use o catálogo: [huggingface.co/models](https://huggingface.co/models).
3. Ele **injeta** adaptadores **LoRA** em camadas específicas da rede (só uma fração dos pesos é treinável).
4. O **TRL** (`SFTTrainer`) faz o treino supervisionado de texto: o modelo aprende a prever o próximo token nas sequências montadas pelo *chat template*.
5. O resultado salvo é um **adapter** (pastas pequenas) + **tokenizer** — **não** uma cópia inteira do modelo base.
6. Na hora de testar, você carrega **o mesmo modelo base** + **o adapter** com o script **`inferir.py`** (detalhes em [README_INFERIR.md](trein/README_INFERIR.md)).
7. **(Opcional)** Depois do treino, `merge_lora.py` **incorpora** o LoRA ao modelo base e grava um **único modelo** em disco (~mesmo tamanho do base no Hub), útil para deploy ou upload sem depender do adapter.

**Por que LoRA?** Em vez de atualizar todos os bilhões de parâmetros do modelo, você treina matrizes pequenas “ao lado” das camadas. Isso **reduz memória** e **tempo**, ideal para notebook sem GPU.

**Por que fundir é opcional?** Com adapter separado você economiza disco e pode trocar adapters sem duplicar gigabytes. Fundir **duplica** o tamanho (modelo completo na pasta de saída), mas simplifica o uso: basta `AutoModelForCausalLM.from_pretrained("pasta/do/merge")` sem PEFT.

---

## Parte A — Instalação (do zero)

### A.1 Criar ambiente virtual

| Passo | Comando | Para que serve |
|-------|---------|----------------|
| 1 | `python3 -m venv .venv` | Cria a pasta `.venv` com um Python isolado só deste projeto. Evita conflitar com outros pacotes do sistema. |

#### O que é a pasta `.venv` (ambiente virtual)

A pasta **`.venv`** guarda o **ambiente virtual** deste projeto: uma cópia “privada” do interpretador Python e de tudo o que você instala com `pip` quando o ambiente está **ativado**.

| Aspecto | Explicação |
|---------|------------|
| **Isolamento** | Bibliotecas como `torch` e `transformers` ficam **dentro** de `.venv`, separadas do Python do sistema e de **outros projetos**. Assim, versões diferentes em cada projeto não se misturam. |
| **Quem usa** | Depois de `source .venv/bin/activate` (Linux/macOS), os comandos `python` e `pip` passam a apontar para o que está em `.venv`. |
| **Tamanho** | Pode ficar **grande** (vários GB) por causa do PyTorch e dependências. Isso é normal. |
| **Git** | O projeto costuma **ignorar** `.venv` no `.gitignore` — não é obrigatório versionar essa pasta; outra pessoa recria com `python3 -m venv .venv` e `pip install -r trein/requirements.txt` (só treino) ou `pip install -r requirements.txt` (treino+servidor). |
| **Apagar** | Se você apagar `.venv`, não perde o código do projeto; só o ambiente. Recrie com os comandos da Parte A e reinstale o ficheiro `requirements` que usar. |

O nome `.venv` é só uma convenção (poderia ser `venv` ou outro nome); o importante é **sempre ativar** esse ambiente antes de rodar os scripts desta pipeline.

### A.2 Ativar o ambiente

| Sistema | Comando |
|---------|---------|
| Linux / macOS | `source .venv/bin/activate` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

Enquanto estiver ativo, o terminal costuma mostrar `(.venv)` no início da linha. **Sempre ative** antes de rodar `train_lora.py`, `inferir.py` ou `merge_lora.py`.

### A.3 Instalar dependências

Só a pipeline de **treino** (recomendado para esta guia, sem API nem PostgreSQL):

```bash
python -m pip install --upgrade pip
pip install -r trein/requirements.txt
```

Se quiser **máquina de desenvolvimento** com Oráculo + treino, na raiz: `pip install -r requirements.txt` (inclui `trein/` e `server/`).

| Pacote (resumo) | Função |
|-----------------|--------|
| `torch` | Tensores e treino da rede. |
| `transformers` | Carregar modelo e tokenizer do Hub. |
| `datasets` | Ler JSONL e preparar batches. |
| `accelerate` | Infraestrutura de treino (usada pelo Trainer). |
| `peft` | LoRA e carregamento de adapters. |
| `trl` | `SFTTrainer` e `SFTConfig` para treino no formato texto. |
| `sentencepiece` / `protobuf` | Muitos tokenizers dos modelos precisam disso. |

### A.3.1 Qual `requirements` usar: normal vs quantizado (AWQ)

**O que é “peso quantizado”**  
No treino usual, os pesos da rede são números em **precisão alta** (por exemplo **FP16** ou **BF16**, 16 bits por valor). **Quantizar** significa **representar cada peso com menos bits** (por exemplo **4 bits**) através de um esquema de codificação. O ficheiro no disco fica **menor** e a inferência pode usar **menos VRAM**, mas os valores já **não** são os pesos originais em ponto flutuante completo — foram **comprimidos** de forma aproximada.

**O que é AWQ e “4 bits”**  
**AWQ** (*Activation-aware Weight Quantization*) é um **método** de quantização. Repositórios marcados como **`-AWQ`** no Hugging Face (muitos publicados por perfis como **TheBloke**) são **checkpoints já convertidos** para esse formato, em geral **4 bits por peso** (daí “menos memória, peso aproximado”). O modelo continua a ser **só texto** no sentido de tarefa (gerar código/respostas); o que muda é o **formato numérico dos tensores** guardados, não o facto de ser multimodal.

**Por que `TheBloke/deepseek-coder-1.3b-instruct-AWQ` é AWQ**  
Esse ID aponta para um repositório onde os pesos **já vêm quantizados** (AWQ). Por isso o `transformers` **não** carrega só com PyTorch “genérico”: precisa de bibliotecas extra (**`gptqmodel`**, e para LoRA sobre essas camadas também **`optimum`**, ver [trein/requirements-quantized.txt](trein/requirements-quantized.txt)). Já um ID como **`deepseek-ai/deepseek-coder-1.3b-instruct`** (sem sufixo AWQ) costuma trazer pesos em **FP16/BF16** “normais” — basta o [trein/requirements.txt](trein/requirements.txt) habitual.

**Quando usar cada ficheiro**

| Situação | O que instalar |
|----------|----------------|
| Modelo base **não quantizado** no Hub (FP16/BF16, ou o cartão **não** menciona AWQ/GPTQ/4-bit) — ex.: `TinyLlama/...`, `deepseek-ai/deepseek-coder-1.3b-instruct` | `pip install -r trein/requirements.txt` |
| Modelo **AWQ** ou **GPTQ** no Hub — nome ou cartão com **AWQ**, **4-bit**, **TheBloke/...-AWQ**, etc. | `pip install -r trein/requirements.txt` **e** `pip install -r trein/requirements-quantized.txt` |

**Recomendação para esta pipeline de LoRA**  
Para **aprender e treinar com menos surpresas**, prefira um modelo **sem quantização** (`deepseek-ai/...`, `TinyLlama/...`, etc.). Use checkpoints **AWQ** em geral quando precisar de **caber em menos VRAM** na **inferência** ou quando já domina o ambiente; o treino LoRA em cima de AWQ é **mais exigente** em dependências e pode emitir avisos sobre tipos de camada.

### A.4 Verificar instalação

```bash
python trein/verificar_ambiente.py
```

| Saída | Significado |
|-------|-------------|
| `CUDA disponível: False` | Normal em notebook **sem** GPU NVIDIA; o treino usa **CPU**. |
| `CUDA disponível: True` | Há GPU; o script usará **bf16/fp16** conforme suporte. |

### A.5 Onde pesquisar modelos no Hugging Face Hub

O site oficial para **listar e filtrar** modelos públicos (causais, chat, tamanho, licença, downloads, etc.) é:

**[https://huggingface.co/models](https://huggingface.co/models)**

O identificador que aparece nos scripts (`data_config.py` → `DEFAULT_MODEL_NAME`, ou `--model_name` no treino/inferência/merge) é o **nome do repositório** no Hub, no formato `organização/nome-do-modelo` (ex.: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`). Prefira variantes **Chat** ou **Instruct** quando o treino usar `messages` e `apply_chat_template`.

---

## Parte B — Preparar os dados

### B.1 Formato do arquivo

- **Tipo:** JSONL (uma linha = um objeto JSON).
- **Coluna obrigatória:** `messages`.

Cada elemento de `messages` é um objeto com:

| Campo | Valores típicos | Função |
|-------|-----------------|--------|
| `role` | `"user"`, `"assistant"` (e às vezes `"system"`) | Quem “fala” naquela parte do diálogo. |
| `content` | texto | O que foi dito. |

**Regra prática:** cada linha deve conter pelo menos um turno de **usuário** e um de **assistente** com a resposta que você quer que o modelo aprenda a imitar.

Exemplo mínimo (uma linha no arquivo):

```json
{"messages": [{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Olá! Em que posso ajudar?"}]}
```

Se você tem **muitas fontes** (conversas, livros, Bíblia, psicologia, sintéticos, etc.) e o volume cresce com o tempo, veja o guia **[README_DATASETS.md](trein/README_DATASETS.md)** — pastas `raw/` e `snapshots/`, metadados, balanceamento e quando unificar ou separar treinos.

### B.2 O que o script faz com esses dados (internamente)

1. **`load_dataset("json", ...)`** — lê todas as linhas em um conjunto chamado `train`.
2. Para cada exemplo, **`apply_chat_template(..., tokenize=False)`** — transforma `messages` em **um único texto** no formato que o modelo espera (marcadores `<|user|>`, `<|assistant|>`, etc., dependendo do modelo).
3. Esse texto vai para a coluna **`text`**, que é o que o `SFTTrainer` usa (`dataset_text_field="text"`).
4. O tokenizer **tokeniza** e **trunca** para no máximo **`max_length`** (no script: argumento `--max_seq_length`).

---

## Parte C — Treino: `train_lora.py`

### C.1 Comando básico

Com **snapshot** gerado a partir de `trein/data/raw/` (recomendado quando já há dados organizados):

```bash
python trein/build_snapshot.py
python trein/train_lora.py --output_dir trein/outputs/lora_adapter
```

Sem `--train_file`, o treino usa o arquivo `train_*_v*.jsonl` de **maior versão** em `trein/data/snapshots/` (ver `trein/data_config.py`). Detalhes da lógica: [README_SNAPSHOT.md](trein/README_SNAPSHOT.md); organização dos dados: [README_DATASETS.md](trein/README_DATASETS.md).

Forçar um JSONL concreto (ex.: teste rápido):

```bash
python trein/train_lora.py --train_file trein/data/raw/exemplo/exemplo_treino.jsonl --output_dir trein/outputs/lora_adapter
```

Se você não passar outros argumentos, valem os **padrões** descritos na tabela abaixo.

### C.2 Argumentos da linha de comando (todos)

| Argumento | Padrão | O que faz |
|-----------|--------|-----------|
| `--model_name` | `data_config.DEFAULT_MODEL_NAME` | **ID do modelo** no Hugging Face Hub (definido em [data_config.py](trein/data_config.py)). O mesmo ID deve ser usado na inferência ([README_INFERIR.md](trein/README_INFERIR.md)) e no merge. Prefira variantes **Chat** ou **Instruct** para ter `apply_chat_template` correto. |
| `--train_file` | (omitido) | **Caminho** do JSONL com a coluna `messages`. Se **omitido**: maior versão em `trein/data/snapshots/`, senão `trein/data/raw/exemplo/exemplo_treino.jsonl` se existir. Edite `trein/data_config.py` e rode `python trein/build_snapshot.py` para gerar o snapshot. |
| `--output_dir` | `data_config.DEFAULT_ADAPTER_DIR` | **Pasta** onde serão salvos o adapter LoRA, checkpoints e o tokenizer (valor em [data_config.py](trein/data_config.py)). |
| `--epochs` | `3.0` | Quantas **vezes** o algoritmo passa por todo o dataset de treino. Valores fracionários são permitidos (ex.: `0.5` = meia época). |
| `--max_steps` | `-1` | Se for **maior que zero**, o treino **para após N atualizações** (passos), em vez de depender só de `--epochs`. Útil para **teste rápido** (`1` passo) ou para limitar tempo. Quando `> 0`, o Trainer do Hugging Face usa esse limite de passos. |
| `--lr` | `2e-4` | **Taxa de aprendizado**: quão grandes são os ajustes nos pesos LoRA a cada passo. Muito alto pode destabilizar; muito baixo pode aprender devagar. `2e-4` é um ponto de partida comum para LoRA. |
| `--batch_size` | `1` | **Quantos exemplos** entram na memória **por vez** em cada dispositivo (aqui, normalmente CPU inteira = 1). Em 16 GB de RAM, **1** é o mais seguro. |
| `--grad_accum` | `8` | **Acúmulo de gradiente**: o otimizador só atualiza os pesos a cada `grad_accum` micro-batches. O **batch lógico** fica próximo de `batch_size × grad_accum` (ex.: 1×8 = 8). Aumenta estabilidade sem multiplicar a RAM do batch. |
| `--max_seq_length` | `512` | **Comprimento máximo** de cada sequência em **tokens** após tokenização. Textos maiores são **cortados**. Reduzir (ex.: `256`) **economiza RAM**; aumentar exige mais memória. |
| `--lora_r` | `8` | **Posto (“rank”)** das matrizes LoRA: capacidade do adapter. Valores maiores = mais parâmetros treináveis e mais risco de overfitting/memória. `8` é um começo equilibrado. |
| `--lora_alpha` | `16` | **Escala** do LoRA em relação a `r`. Na prática, costuma ser `2× r` (ex.: r=8 → alpha=16). Afeta a **magnitude** da adaptação. |
| `--no_gradient_checkpointing` | (desligado) | Se **não** passar essa flag, o **gradient checkpointing** fica **ligado**: recalcula ativações no backward para **usar menos RAM**, trocando por um pouco mais de tempo. Passe a flag se tiver RAM sobrando e quiser tentar acelerar um pouco. |
| `--trust_remote_code` | (desligado) | Alguns modelos executam código customizado do Hub. Só use se o cartão do modelo no Hugging Face pedir e você confiar na origem. |

**Ajuda no terminal:**

```bash
python trein/train_lora.py --help
```

### C.3 Configurações fixas no código (não são argumentos)

Estas aparecem em `train_lora.py` e só mudam se você editar o arquivo:

| Configuração | Valor usado | Para que serve |
|--------------|-------------|----------------|
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | **Onde** o LoRA é aplicado (arquitetura estilo **Llama/TinyLlama**). Outros modelos podem ter nomes diferentes. |
| `lora_dropout` | `0.05` | **Dropout** nos adaptadores LoRA: ajuda a generalizar; 0.05 é moderado. |
| `bias` | `"none"` | Não treina bias extra no LoRA (padrão PEFT comum). |
| `task_type` | `CAUSAL_LM` | Indica modelo causal (próximo token). |
| `warmup_ratio` | `0.03` | **3%** dos passos totais com learning rate subindo do zero (aquecimento). |
| `lr_scheduler_type` | `cosine` | Learning rate varia em forma de **coseno** ao longo do treino. |
| `logging_steps` | `5` | A cada **5** passos imprime métricas no console. |
| `save_steps` | `200` | A cada **200** passos salva checkpoint na `--output_dir`. |
| `save_total_limit` | `2` | Mantém só os **2** checkpoints mais recentes (economiza disco). |
| `report_to` | `"none"` | **Não** envia logs para Weights & Biases nem TensorBoard automaticamente. |
| `packing` | `False` | Não agrupa vários exemplos numa mesma sequência (mais simples e previsível para começar). |
| `dataset_text_field` | `"text"` | Nome da coluna com o texto já formatado pelo chat template. |
| `optim` | `adamw_torch` em CPU, `adamw_torch_fused` em GPU | Otimizador; a variante **fused** costuma ser só para CUDA. |
| `dataloader_pin_memory` | `True` só com GPU | Acelera transferência CPU→GPU quando há CUDA. |
| `use_cpu` | `True` se não houver CUDA | Obrigatório para o Trainer aceitar treino em CPU nas versões recentes do Hugging Face. |

### C.3.1 Métricas no console: loss, bias e LoRA

É fácil confundir **nomes parecidos** com **indicadores de qualidade**. No treino desta pipeline vale o seguinte:

| Ideia | Explicação |
|-------|------------|
| **Bias** (vetor de *bias* nas camadas lineares) | É um **parâmetro** que a rede aprende, como os pesos. **Não** é uma métrica do tipo “quanto mais perto de zero, melhor o treino”. O valor ideal depende da camada e do problema. |
| **Loss** (`loss` no log) | É o indicador **numérico principal**: erro de previsão do próximo token (cross-entropy). **Loss mais baixa** no conjunto de **treino** significa melhor encaixe nesses dados — mas, se for a única métrica, pode mascarar **overfitting** (decorar o dataset). |
| **LoRA** | Não existe uma variável exclusiva de LoRA que substitua a loss. O que acompanhas é o **mesmo tipo de sinal** que num fine-tuning completo: sobretudo **training loss**; se no futuro adicionares **validação**, **validation loss** ajuda a ver generalização. |
| **`lora_r`, `lora_alpha`, `lora_dropout`** | São **hiperparâmetros** definidos **antes** do treino; não são “painéis” ao vivo de qualidade. |
| **`print_trainable_parameters()`** (no início do treino) | Mostra **quantos** parâmetros são treináveis — útil para tamanho do adapter, **não** para julgar se o treino “está bom”. |

**Onde ver a loss:** com `logging_steps` = `5`, o `SFTTrainer` imprime métricas **a cada 5 passos** no terminal (`train_loss`, etc., conforme a versão do TRL). O que é **passo** vs **época** e por que a barra mostra `X/Y` em passos — **§C.3.2**; como ler cada campo do log — **§C.3.3**.

**Perplexidade (opcional):** em linguagem de modelos de linguagem, **perplexidade** ≈ \(\exp(\text{loss})\) (na mesma base do log). Serve só como **outra forma de ler** o mesmo número — não é obrigatório acompanhar.

**Como interpretar sem conjunto de validação** (como está o script por omissão): use a **curva de loss** (ainda desce? já está plana?) e, sobretudo, **testes reais** com `inferir.py` em perguntas **novas** ou reformuladas. Se a loss baixar muito mas o modelo **copiar** frases do JSONL ou **piorar** fora do treino, pode ser época ou learning rate em excesso — ajuste `--epochs`, `--lr` e o tamanho do dataset antes de confiar só no número da loss.

### C.3.2 Época, passo e `logging_steps` (barra `X/Y`)

Três ideias **separadas**:

| Conceito | O que é |
|----------|---------|
| **Época (*epoch*)** | **Uma volta completa** pelo dataset de treino: o algoritmo usa **todas** as linhas do JSONL **uma vez** (por ordem ou embaralhadas, conforme o *dataloader*). Com `--epochs 6`, isso repete **seis vezes** — volta ao início do dataset após cada época. |
| **Passo (*step*)** | **Uma atualização dos pesos** LoRA: o otimizador faz **um** `optimizer.step()` após processar um ou mais *batches* e, se `gradient_accumulation_steps` > 1, acumular gradientes. **Não** é “uma linha do JSONL” — é **um** ajuste da rede. |
| **`logging_steps` (fixo em `5` no código)** | **Só** a frequência com que o terminal **imprime** métricas: a cada **5 passos** (5 atualizações), não a cada 5 épocas nem a cada 5 linhas. **Não** altera o que é treinado. |

**Barra de progresso `7/30`:** o Trainer mostra **passo atual / total de passos** do treino inteiro (todas as épocas incluídas). O total depende do tamanho do dataset, `batch_size`, `grad_accum` e `epochs`. **Não** é “época 1 de 6” — por isso não verá `1/6`, `2/6`… na barra padrão. O campo `'epoch'` dentro de cada linha de log indica **aproximação** da época em que o passo cai.

**O que *não* é verdade:**

- **Não** é “cada linha do JSONL executa 5 passos”.
- **Não** é “contar 5 voltas ao dataset para fechar uma época”.
- **`logging_steps=5`** não define a ordem em que os exemplos são lidos — só **quando** o texto do log aparece.

**Fluxo correto (resumo):** em cada época o loader **varre o dataset em batches** até cobrir **todos** os exemplos; isso gera **vários** passos por época. Depois começa a época seguinte. O **5** regula apenas a **impressão** das métricas no ecrã.

### C.3.3 Ler o log do treino (exemplo)

Exemplo típico de saída (números ilustrativos):

```text
30/30 [24:55<00:00, 49.26s/it]
{'loss': '5.325', 'grad_norm': '0.3003', 'learning_rate': '0.0001948', 'entropy': '3.321',
 'num_tokens': '6661', 'mean_token_accuracy': '0.272', 'epoch': '1'}
```

| Parte do log | Significado |
|--------------|-------------|
| **`30/30`** | Acabaram **30 passos** de otimização no total do run (não “30 épocas”). |
| **`49.26s/it`** | Tempo médio por **passo** (*iteration*); em CPU costuma ser alto. |
| **`logging_steps=5`** | Com seis blocos de métricas para 30 passos, costuma haver log nos passos 5, 10, 15, …, 30. |
| **`'epoch': '1'` … `'6'`** | Progresso aproximado da **época** ao longo do treino (alinhado ao teu `--epochs` quando o total de passos fecha as voltas ao dataset). |

**Campos do dicionário:**

| Campo | Leitura rápida |
|-------|----------------|
| **`loss`** | Erro de próximo token (cross-entropy) no **treino**. **Descer** ao longo do tempo costuma indicar que o modelo **encaixa melhor** nesses dados. |
| **`grad_norm`** | Norma do gradiente. Valores **estáveis** (sem explosões) são bons sinais; picos estranhos ou NaN indicam instabilidade. |
| **`learning_rate`** | Taxa efetiva no passo. Com `lr_scheduler_type=cosine`, tende a **cair** do valor inicial quase até zero no fim do treino — é **esperado**. |
| **`entropy`** | Contexto da distribuição de tokens no batch; útil como complemento, não como único critério de “sucesso”. |
| **`num_tokens`** | **Tokens processados** no acumulado (sobe ao longo do treino). |
| **`mean_token_accuracy`** | Fração de tokens em que o modelo acertou o próximo token **no treino**. **Subir** indica melhor encaixe nos dados vistos; não prova generalização. |

**Como julgar se “o aprendizado foi bom” (no treino):**

- **Sinais favoráveis:** `loss` desce de forma consistente; `mean_token_accuracy` sobe; `grad_norm` sem divergência.
- **Limite:** isto avalia só o **ajuste ao dataset de treino**. **Não** substitui testar com **`inferir.py`** em perguntas **novas** ou fora do JSONL.
- **Alerta:** se a loss fica muito baixa mas o modelo **repete** trechos do ficheiro ou **piora** fora do treino, pode ser **overfitting** — ajuste dados, épocas ou learning rate.

Ordens de grandeza da `loss` dependem do modelo e do dataset; o importante é a **tendência** e o comportamento na **inferência**.

### C.4 O que acontece na ordem (fluxo interno)

1. Carrega **tokenizer** do `--model_name`.
2. Ajusta **pad_token** se o modelo não tiver (usa `eos`).
3. Lê o **JSONL** e valida a coluna **`messages`**.
4. Converte cada linha em **`text`** via **`apply_chat_template`**.
5. Carrega o **modelo base** (fp32 em CPU; bf16/fp16 em GPU quando aplicável).
6. Envolve o modelo com **LoRA** (`get_peft_model`) e imprime quantos parâmetros são treináveis.
7. Monta **`SFTConfig`** com seus argumentos + opções fixas.
8. Cria **`SFTTrainer`** e chama **`train()`**.
9. **`save_model`** grava adapter + **`tokenizer.save_pretrained`** na `--output_dir**.

### C.5 Exemplos de comandos

**Só testar se a máquina aguenta (1 passo):**

```bash
python trein/train_lora.py --max_steps 1 --output_dir trein/outputs/teste_instalacao --train_file trein/data/raw/exemplo/exemplo_treino.jsonl
```

**Treino mais longo com menos RAM por sequência:**

```bash
python trein/train_lora.py \
  --train_file trein/data/meu_dataset.jsonl \
  --output_dir trein/outputs/lora_adapter \
  --epochs 5 \
  --max_seq_length 256 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-4
```

**Outro modelo pequeno (pode exigir código confiável no Hub):**

```bash
python trein/train_lora.py \
  --model_name "Qwen/Qwen2-0.5B-Instruct" \
  --trust_remote_code \
  --train_file trein/data/meu_dataset.jsonl
```

### C.6 Arquivos gerados em `--output_dir`

| Tipo | Descrição |
|------|-----------|
| Pesos LoRA (`adapter_model.safetensors` ou `.bin`, etc.) | Só os **adaptadores** treinados. |
| `adapter_config.json` | Descreve rank, alpha, módulos alvo. |
| Tokenizer | Mesmos arquivos que o Hugging Face costuma salvar (`tokenizer.json`, etc.). |
| Checkpoints opcionais | Pastas `checkpoint-*` se o treino passar de `save_steps`. |
| Modelo **fundido** (opcional, após `merge_lora.py`) | Pasta `--output_dir` do merge: pesos do modelo **completos** com LoRA incorporado (~mesma ordem de tamanho do base no Hub) + tokenizer. Ver **Parte E**. |

O **modelo base** continua sendo baixado do Hub na inferência; você **não** duplica os ~1B de parâmetros base no adapter. **Onde esse modelo base fica no disco** e **como listar** o que já foi baixado está na **Parte H**.

---

## Parte D — Teste de inferência (`inferir.py`)

Depois do treino, use **`inferir.py`** para carregar modelo base + adapter LoRA e gerar uma resposta no terminal. **Lógica interna, argumentos, parâmetros de geração e limitações:** [README_INFERIR.md](trein/README_INFERIR.md).

```bash
python trein/inferir.py --adapter_dir trein/outputs/lora_adapter --prompt "Sua pergunta aqui"
```

---

## Parte E — Fundir adapter + base: `merge_lora.py`

### E.1 O que é “fundir” (merge)

Durante o treino, o LoRA fica em **arquivos pequenos** separados do modelo base. O **merge** usa o PEFT para **somar** essas atualizações aos pesos do base e produzir um **único** `AutoModelForCausalLM` “normal”, que você salva com `save_pretrained`.

| Situação | Adapter separado (teste com [README_INFERIR.md](trein/README_INFERIR.md)) | Modelo fundido (`merge_lora.py`) |
|----------|----------------------------------|----------------------------------|
| Tamanho em disco (exemplo TinyLlama) | Poucos MB na pasta do adapter + ~2,2 GB **só no cache** do Hub | ~2,2 GB **na pasta** `--output_dir` do merge (cópia completa) |
| Carregar na inferência | Base + `PeftModel.from_pretrained` | Só `from_pretrained` na pasta fundida |
| Quando preferir | Dia a dia, testes, vários adapters | Entregar um pacote único, conversão GGUF, upload como modelo completo |

**Momento:** **depois** do `train_lora.py`, quando você já tem `adapter_config.json` e os pesos LoRA na pasta do adapter. **Não** substitui o treino; é um passo **opcional** pós-treino.

**RAM:** em CPU o script carrega o modelo base inteiro (ex.: fp32). Para ~1,1B parâmetros, reserve **vários GB de RAM livres**; se faltar memória, use máquina com mais RAM ou GPU.

### E.2 Comando

```bash
python trein/merge_lora.py \
  --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --adapter_dir trein/outputs/lora_adapter \
  --output_dir trein/outputs/merged_model
```

### E.3 Argumentos

| Argumento | Padrão | O que faz |
|-----------|--------|-----------|
| `--model_name` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | **Mesmo** ID do modelo base usado no treino. O merge baixa/carrega esses pesos do Hub (ou do cache). |
| `--adapter_dir` | `trein/outputs/lora_adapter` | Pasta com `adapter_config.json` + pesos LoRA (saída do `train_lora.py`). |
| `--output_dir` | `trein/outputs/merged_model` | Onde gravar o modelo causal **fundido** + tokenizer. |
| `--trust_remote_code` | (desligado) | Igual aos outros scripts, se o modelo base exigir. |

O tokenizer é lido da pasta do adapter se existir `tokenizer_config.json`; senão, usa o do `--model_name`.

### E.4 Depois do merge

Para gerar texto com o modelo fundido em Python (sem PEFT):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "trein/outputs/merged_model"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)  # ou bf16/fp16 com GPU
# ... apply_chat_template + generate (lógica semelhante ao README_INFERIR.md) ...
```

---

## Parte F — Checklist rápido

| Ordem | Ação |
|-------|------|
| 1 | Ativar `.venv` e instalar `trein/requirements.txt` (ou `requirements.txt` na raiz para treino+servidor). |
| 2 | Rodar `python trein/verificar_ambiente.py`. |
| 3 | Colocar `.jsonl` em `trein/data/raw/` (por tema) e rodar `python trein/build_snapshot.py` para gerar `trein/data/snapshots/train_*_v*.jsonl` (ou usar `trein/data/raw/exemplo/exemplo_treino.jsonl` só para teste). |
| 4 | Rodar `python trein/train_lora.py --max_steps 1` para validar (sem `--train_file` usa o snapshot de maior versão). |
| 5 | Rodar treino real com `--epochs` / `--max_seq_length` desejados. |
| 6 | Testar inferência: [README_INFERIR.md](trein/README_INFERIR.md) — `inferir.py` com o mesmo `--model_name` do treino. |
| 7 | (Opcional) Fundir base + adapter: `python trein/merge_lora.py --adapter_dir trein/outputs/lora_adapter --output_dir trein/outputs/merged_model` (Parte E). |
| 8 | (Opcional) Ver onde o modelo base foi cacheado: Parte H — `hf cache scan` ou `ls ~/.cache/huggingface/hub/`. |

---

## Parte G — Problemas comuns

| Sintoma | O que tentar |
|---------|----------------|
| Erro de memória (RAM) | Reduzir `--max_seq_length`; manter `--batch_size 1`; não usar `--no_gradient_checkpointing`. |
| Treino muito lento | Esperado em **CPU**; reduzir dados, épocas ou `max_steps` para experimentos. |
| Respostas ruins | Mais exemplos de qualidade no JSONL; ajustar `--epochs`, `--lr`, `--lora_r`; na inferência de teste, o mesmo `model_name` do treino ([README_INFERIR.md](trein/README_INFERIR.md)). |
| `trust_remote_code` pedido | Adicionar `--trust_remote_code` no treino, na inferência e no `merge_lora.py` para aquele modelo. |
| Erro de memória no **merge** | O `merge_lora.py` carrega o modelo base inteiro; feche outros programas, use GPU se disponível, ou funda em máquina com mais RAM. |
| `ValueError: Unrecognized configuration class ... JanusConfig` for `AutoModelForCausalLM` | O modelo no Hub **não** é um causal LM “só texto” compatível com esta pipeline (ex.: **Janus** é multimodal). Escolha outro ID em [huggingface.co/models](https://huggingface.co/models) (filtro *Text Generation* / variantes **Chat** ou **Instruct**) e atualize `DEFAULT_MODEL_NAME` em [data_config.py](trein/data_config.py) ou passe `--model_name`. |
| `ImportError: Loading an AWQ quantized model requires gptqmodel` | Checkpoints **AWQ** precisam de `pip install -r trein/requirements-quantized.txt` (ver [trein/requirements-quantized.txt](trein/requirements-quantized.txt)). **Contexto:** o que são pesos quantizados e quando usar cada `requirements` — **§A.3.1**. **Alternativa:** modelo **sem** quantização (ex.: `deepseek-ai/deepseek-coder-1.3b-instruct`) e só o [trein/requirements.txt](trein/requirements.txt). |

---

## Parte H — Cache do Hugging Face: modelo base e o que foi baixado

Esta seção responde: **onde o modelo original (base) fica salvo** no computador e **como listar** o que já foi baixado pelo `transformers` / Hub.

### H.1 O que fica onde

| O quê | Onde fica |
|-------|-----------|
| **Pesos do modelo base** (ex.: TinyLlama baixado do Hub) | **Cache global** do Hugging Face — em Linux, em geral `~/.cache/huggingface/hub/`. **Não** fica dentro da pasta `trein/outputs/` do projeto. |
| **Adapter LoRA** e cópia do tokenizer do seu treino | Pasta `--output_dir` (ex.: `trein/outputs/lora_adapter`). |

Na primeira vez que você roda `train_lora.py`, `inferir.py` ou `merge_lora.py`, a biblioteca **baixa** (ou reutiliza) o modelo base nesse cache. O treino **não** grava uma segunda cópia completa do modelo base na pasta do adapter — só o LoRA. O **merge** grava uma cópia **completa** fundida em `--output_dir` do `merge_lora.py` (Parte E).

### H.2 Mudar o diretório de cache (opcional)

Você pode redirecionar onde o Hugging Face guarda dados com variáveis de ambiente, por exemplo:

| Variável | Efeito (resumo) |
|----------|------------------|
| **`HF_HOME`** | Diretório “raiz” do Hugging Face; o cache de modelos costuma ficar em `HF_HOME/hub`. |
| **`HF_HUB_CACHE`** | Em versões recentes, aponta diretamente para a pasta do cache de repositórios (quando definida). |

Se não configurar nada, o padrão no Linux costuma ser `~/.cache/huggingface/`.

download hg
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.6-35B-A3B --max-workers 1

### H.3 Ver qual caminho o sistema está usando

Com a CLI do Hub instalada (vem com `huggingface_hub`, normalmente junto do ecossistema Transformers), use a CLI unificada **`hf`**:

```bash
hf env
```

Em versões mais antigas ainda existe `huggingface-cli env` (equivalente). Procure linhas como `HF_HOME`, `HF_HUB_CACHE` ou `CACHE` para ver os caminhos efetivos.

### H.4 Listar modelos / repositórios já baixados

**Forma recomendada (CLI):**

```bash
hf cache scan
```

Mostra repositórios presentes no cache, revisões e tamanhos (o detalhe depende da versão da CLI). O comando antigo `huggingface-cli scan-cache` está **obsoleto**; use `hf cache scan` em vez dele.

**Listar só as pastas no disco (Linux):**

```bash
ls ~/.cache/huggingface/hub/
```

Os nomes das pastas seguem um padrão do tipo `models--NomeOrg--NomeDoRepo` (o `--` substitui o `/` do ID `NomeOrg/NomeDoRepo` no Hub).

**Por código Python** (útil para scripts):

```python
from huggingface_hub import scan_cache_dir

cache = scan_cache_dir()
for repo in cache.repos:
    print(repo.repo_id, repo.size_on_disk_str)
```

Se a sua versão de `huggingface_hub` tiver API ligeiramente diferente, consulte a documentação oficial de `scan_cache_dir` na versão instalada.

### H.5 Se você apagar o cache

Se remover arquivos dentro de `hub/` (ou o cache inteiro), na próxima execução de `train_lora.py`, `inferir.py` ou `merge_lora.py` o modelo base será **baixado de novo** (internet e acesso ao Hub necessários), desde que o repositório continue público e disponível.

---

## Glossário

| Termo | Significado |
|-------|-------------|
| **Token** | Pedaço de texto (palavra, sílaba ou símbolo) que o tokenizer converte em número. |
| **Época (epoch)** | Uma passagem completa pelo dataset de treino. |
| **Passo (step)** | Uma atualização do otimizador (após `grad_accum` micro-batches, conforme configuração). |
| **LoRA** | Low-Rank Adaptation: matrizes pequenas adicionadas às camadas escolhidas. |
| **SFT** | Supervised Fine-Tuning: treino em texto supervisionado (aqui, conversas). |
| **Adapter** | Pasta com só os pesos LoRA; precisa do modelo base para funcionar. |
| **Merge / modelo fundido** | Pesos do base com LoRA já incorporados; um único `from_pretrained` na pasta gerada pelo `merge_lora.py`. |

---

Em **CPU**, o gargalo é tempo, não apenas “possível ou não”: com dataset pequeno e TinyLlama, esta pipeline serve para **aprender**, **testar formato de dados** e **validar ideias** antes de escalar para GPU ou nuvem. O passo **merge** também exige **RAM** confortável para carregar o modelo base inteiro.
