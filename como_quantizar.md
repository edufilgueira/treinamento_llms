https://github.com/ggerganov/llama.cpp



# levantar servidor llama.cpp

```bash
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build -j
```


```bash
# matar serviço
pgrep -a llama-server
kill -9 <PID>

#  levantar servidor
./build/bin/llama-server \
  -m "$HOME/treinamento_llms/tools/quantized_model/Qwen3-8B-base-F16-Q4_K_M.gguf" \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  --reasoning off \
  --reasoning-budget 0
```



```bash
apt install zip -y
# compactar no servidor
zip -r -s 1g merged_qwen3-8B.zip merged_qwen3-8B
# descompactar
sudo apt install p7zip-full -y
7z x merged_qwen3-8B.zip

# download
hf download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF --local-dir ~/models/qwen35b-gguf --max-workers 1

# listar processos
ps -aux | grep hf

# Deletar cache de download
rm -rf ~/models/qwen35b-gguf/.cache/huggingface/download/*.lock
rm -rf ~/.cache/huggingface/hub/.locks/*
```

# Fluxo: copiar GGUF para treinamento_llm/tools
```bash
ls ~/.cache/huggingface/hub/

ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

### Ajusta o caminho ao script e ao modelo, conforme a tua versão de llama.cpp:
## 1) Modelo em repositorio HuggingFace no disco
python ~/treinamento_llms/tools/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/ \
  --outfile ~/treinamento_llms/tools/quantized_model/Qwen3-8B-F16.gguf \
  --outtype f16

## 2) Modelo em mergeado HuggingFace no disco
python ~/treinamento_llms/tools/llama.cpp/convert_hf_to_gguf.py \
  ~/treinamento_llms/trein/outputs/merged_qwen3-8B \
  --outfile ~/treinamento_llms/tools/quantized_model/Qwen3-8B-F16.gguf \
  --outtype f16


cd ~/treinamento_llms/tools/llama.cpp/

./build/bin/llama-quantize \
  ~/treinamento_llms/tools/quantized_model/Qwen3-8B-F16.gguf \
  ~/treinamento_llms/tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf \
  Q4_K_M

```




---

# 🚀 PIPELINE PADRÃO — LoRA (mergeado) → GGUF → Quantizado

Baseado no seu fluxo real usando llama.cpp

---

# 📁 Estrutura do projeto

```bash
treinamento_llms/
├── server/
│   ├── inference/      # runtime HF / GGUF / llama-server
│   ├── api/v1/         # OpenAI-compat
│   └── db/
├── trein/
│   └── outputs/
│       └── merged_model/
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           └── ...
├── tools/
│   ├── llama.cpp/
│   └── quantized_model/
```

---

# 📥 A mesma cadeia, mas o modelo veio do Hugging Face (download em disco)

Há **dois** casos; não confundas:

| O que descarregaste do Hub | O que fazer |
|----------------------------|------------|
| **Pasta de modelo “HF completo”** (`config.json` + `model*.safetensors`, *tokenizer*) | É o **mesmo** input que o `merge_lora` grava. Usas o `convert_hf_to_gguf.py` a apontar para **essa pasta local**. Não interessa se o merge veio de LoRA ou se fizeste `huggingface-cli download` de um *checkpoint* aberto. |
| **Só ficheiros `.gguf`** (repositórios *TheBloke*/*-GGUF*) | **Não** corres `convert_hf_to_gguf` — já é GGUF. Se for um `.gguf` **F16/BF16** “grande”, podes **só** `llama-quantize` para `Q4_K_M` / `Q5_K_M`. Se descarregaste **já** `Q4_K_M` / `Q5_K_M`, usa esse ficheiro no Oráculo; **não** convém “re-quantizar” sem necessidade. |

## 1) Modelo em formato HuggingFace (safetensors) no disco

Instala a CLI (uma vez): `pip install -U "huggingface_hub[cli]"`

Descarrega o snapshot para **uma pasta tua** (exemplo: Mistral 7B instruct):

```bash
mkdir -p ~/hf_models
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir ~/hf_models/Mistral-7B-Instruct-v0.3
```

(Se o fornecedor tiver acesso a termos, aceita o modelo no site e, se for preciso, `huggingface-cli login`.)

Dentro do repositório **llama.cpp** (a pasta onde está `convert_hf_to_gguf.py` — os nomes dos scripts mudam entre versões; vê a pasta `tools/` do teu *clone*):

```bash
cd ~/treinamento_llms/tools/llama.cpp
# Ajusta o caminho ao script e ao modelo, conforme a tua versão de llama.cpp:
python tools/convert_hf_to_gguf.py ~/hf_models/Mistral-7B-Instruct-v0.3 --outfile ~/treinamento_llms/tools/quantized_model/Mistral-7B-F16.gguf
```

Notas:

* Em alguns *commits* o script fica noutro sítio (ex. raiz) ou chama-se com **tipo de modelo** extra (`Qwen2`, `Llama`…). Corre `python .../convert_hf_to_gguf.py -h` na tua versão.
* A saída costuma ser **um** `.gguf` (F16/BF32); ajusta `--outfile` para `tools/quantized_model/` se quiseres tudo alinhado ao resto do projecto.

Depois, **igual** ao teu processo: quantizas a partir do **.gguf F16** (ver ETAPA 4 abaixo) e usas `ORACULO_GGUF_PATH` nesse ficheiro final.

## 2) Só ficheiro GGUF (ex. TheBloke)

* Coloca o(s) `.gguf` (ex. `*F16*`, `*BF16*`) em `tools/quantized_model/`.
* **Não** há passo de conversão HF→GGUF.
* Quantização:

```bash
cd ~/treinamento_llms/tools/llama.cpp
./build/bin/llama-quantize \
  ../quantized_model/O-teu-Modelo-F16.gguf \
  ../quantized_model/O-teu-Modelo-Q5_K_M.gguf \
  Q5_K_M
```

O **último** argumento (`Q4_K_M`, `Q5_K_M`, …) tem de coincidir com o sufixo que queres; o ficheiro de saída convém refletir o mesmo (ex. `...-Q5_K_M.gguf` se usaste `Q5_K_M`).

---

# 🧠 ETAPA 1 — Converter para GGUF (modelo “HF” local, ex. merge do projecto)

Entrar no llama.cpp:

```bash
cd ~/treinamento_llms/tools/llama.cpp
```

Rodar conversão **com o caminho que tiveres** (merge **ou** pasta descarregada com `huggingface-cli`):

```bash
python convert_hf_to_gguf.py ../../trein/outputs/merged_model
```

```bash
python3 convert_hf_to_gguf.py \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/ \
  --outfile ~/treinamento_llms/trein/outputs/merged_Qwen3.6-35B-A3B-f16.gguf \
  --outtype f16
```

---

## ✅ Resultado

Arquivo gerado automaticamente em:

```bash
trein/outputs/merged_model/
```

Exemplo:

```bash
Merged_Model-3.1B-BF16.gguf
```

---

# 🧠 ETAPA 2 — Organizar arquivo

Criar pasta de saída:

```bash
mkdir -p ~/treinamento_llms/tools/quantized_model
```

Mover o GGUF:

```bash
mv ~/treinamento_llms/trein/outputs/merged_model/Merged_Model-3.1B-BF16.gguf \
   ~/treinamento_llms/tools/quantized_model/
```

---

# 🧠 ETAPA 3 — Compilar o llama.cpp

Dentro de:

```bash
cd ~/treinamento_llms/tools/llama.cpp
```

Rodar:

```bash
cmake -B build
cmake --build build --config Release
```

---

## ✅ Resultado

Binário gerado em:

```bash
build/bin/llama-quantize
```

---

# 🧠 ETAPA 4 — Quantizar o modelo

Rodar:

```bash
./build/bin/llama-quantize \
../quantized_model/Merged_Model-3.1B-BF16.gguf \
../quantized_model/Merged_Model-3.1B-Q4_K_M.gguf \
Q4_K_M

./build/bin/llama-quantize \
../quantized_model/Merged_Model-3.1B-BF16.gguf \
../quantized_model/Merged_Model-3.1B-Q4_K_M.gguf \
Q4_K_M
```

(O 3.º parâmetro **tem de** corresponder à quantização escolhida; o nome do ficheiro de saída devia bater com isso, ex. `Q5_K_M` no nome **e** no comando se quiseres Q5.)

```bash
# Exemplo com Q5_K_M (nome e tipo alinhados)
./build/bin/llama-quantize \
../quantized_model/Merged_Model-3.1B-BF16.gguf \
../quantized_model/Merged_Model-3.1B-Q5_K_M.gguf \
Q5_K_M
```

---

## ✅ Resultado final

```bash
tools/quantized_model/
├── Merged_Model-3.1B-BF16.gguf
└── Merged_Model-3.1B-Q4_K_M.gguf
```

---

# ⚡ Tipos de quantização recomendados

| Tipo   | Uso                                |
| ------ | ---------------------------------- |
| Q4_K_M | 🔥 padrão (melhor custo/benefício) |
| Q5_K_M | 🧠 mais qualidade                  |
| Q8_0   | 💪 pesado (quase FP16)             |

---

# 🧠 Observações importantes

### ✔ Seu caso (Qwen2.5)

* Conversão detectou BF16 automaticamente
* Totalmente compatível com quantização

---

### ✔ Nome do binário mudou

Use:

```bash
llama-quantize
```

❌ não existe mais `quantize` sozinho

---

### ✔ Separação correta

| Etapa      | Formato     |
| ---------- | ----------- |
| treino     | safetensors |
| inferência | GGUF        |

---

# 🔥 Fluxo completo final

```text
LoRA → merge → HF model
        ↓
convert_hf_to_gguf
        ↓
GGUF (BF16)
        ↓
llama-quantize
        ↓
GGUF quantizado (Q4_K_M)
```

---

# 🧠 Próximo passo lógico

Agora você já pode:

* rodar com `llama-server`
* ou usar o Oráculo (`server/serve_lora.py`) que já expõe `/v1/chat/completions`

---

O servidor já inclui API estilo OpenAI em `server/api/v1/openai.py` (montada em `server/main.py`).



Esses nomes tipo `Q4_K_M`, `Q5_K_M` parecem crípticos, mas na real eles seguem uma lógica bem direta dentro do llama.cpp.

Vou te explicar de forma prática (nível engenharia, não marketing).

---

# 🧠 O que é quantização (rápido)

Modelo original:

* pesos em **FP16 / BF16** (16 bits)

Quantização:

* reduz para **2, 3, 4, 5, 8 bits**
* diminui RAM + acelera CPU
* perde um pouco de qualidade

---

# 🔍 Decodificando `Q4_K_M`

## 1) `Q4`

👉 número de bits por peso

* `Q2` → 2 bits (muito comprimido, baixa qualidade)
* `Q4` → 4 bits (equilíbrio)
* `Q5` → 5 bits (melhor qualidade)
* `Q8` → 8 bits (quase original)

---

## 2) `_K`

👉 tipo de algoritmo de quantização

* `K` = **K-quantization (mais moderno)**
* usa blocos inteligentes (group-wise)
* melhor que os antigos (`Q4_0`, `Q4_1`)

👉 hoje: **sempre use `_K`**

---

## 3) `_M`

👉 variante interna do algoritmo

* `M` = “mixed / medium optimization”
* equilibra:

  * qualidade
  * velocidade
  * uso de memória

---

# 🧠 Tradução direta

```text
Q4_K_M
↓
4 bits + algoritmo moderno + equilíbrio geral
```

---

# ⚖️ Comparação real (o que importa de verdade)

| Tipo   | RAM       | Qualidade | Uso real   |
| ------ | --------- | --------- | ---------- |
| Q2_K   | 🟢 mínimo | 🔴 ruim   | só teste   |
| Q3_K_M | 🟢 baixo  | 🟠 ok     | edge fraco |
| Q4_K_M | 🟡 médio  | 🟢 bom    | 🔥 padrão  |
| Q5_K_M | 🟠 médio+ | 🟢 ótimo  | produção   |
| Q8_0   | 🔴 alto   | 🟢🔥 top  | GPU forte  |

---

# 🔥 O que muda na prática

## Q4_K_M

* roda em pouca RAM
* resposta rápida
* leve perda de precisão

## Q5_K_M

* mais fiel ao modelo original
* um pouco mais pesado
* melhor pra respostas complexas

---

# 🧠 Insight importante (pouca gente entende)

A perda de qualidade NÃO é linear.

Exemplo:

* Q8 → Q5 → quase não muda
* Q5 → Q4 → leve impacto
* Q4 → Q3 → começa a degradar forte
* Q3 → Q2 → quebra geral

👉 por isso Q4_K_M virou padrão

---

# ⚠️ Particularidade do seu caso (Qwen 3B)

Como seu modelo é pequeno (~3B):

* Q4_K_M → já roda muito bem
* Q5_K_M → pode valer a pena (custo baixo, ganho real)

---

# 🧠 Regra prática (decisão rápida)

Se for:

* CPU comum → **Q4_K_M**
* quer qualidade → **Q5_K_M**
* benchmarking → Q8_0
* produção leve → Q4_K_M

---

# 🔥 Resumo sem enrolação

* `Q4` → quantos bits
* `K` → algoritmo moderno
* `M` → ajuste equilibrado

👉 `Q4_K_M = melhor custo-benefício hoje`

---

```bash
pip install -r server/requirements.txt
pip install -r server/requirements-gguf.txt
export ORACULO_INFERENCE_BACKEND=gguf
export ORACULO_GGUF_PATH="$HOME/treinamento_llms/tools/quantized_model/Merged_Model-3.1B-Q4_K_M.gguf"
cd ~/treinamento_llms && python3 server/serve_lora.py
```