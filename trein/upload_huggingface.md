O repositório **não tem um script pronto** de upload — o merge grava um modelo Hugging Face “completo” e você sobe essa pasta manualmente para o Hub.

## 1. Confirme que o merge terminou

A pasta (ex.: `trein/outputs/merged_model` ou `trein/outputs/merged_qwen3-8b`) deve ter algo como:

- `config.json`
- `model*.safetensors` (ou `model.safetensors`)
- `tokenizer.json`, `tokenizer_config.json`, etc.

No projeto, o padrão atual do merge é `merged_qwen3-8b`, não `merged_model`:

```71:72:trein/data_config.py
DEFAULT_ADAPTER_DIR = REPO_ROOT / "trein" / "outputs" / "lora_adapter_qwen3-8b"
DEFAULT_MERGED_MODEL_DIR = REPO_ROOT / "trein" / "outputs" / "merged_qwen3-8b"
```

Use o caminho **real** da sua pasta.

---

## 2. Login no Hugging Face

No Linux, na raiz do repo (com o venv ativo):

```bash
source .venv-trein/bin/activate
pip install -U "huggingface_hub[cli]"
hf auth login
# descobrir usuario logado
hf auth whoami
```

Cole um token com permissão **Write** em: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 3. Crie o repositório no Hub

Pelo site: [https://huggingface.co/new](https://huggingface.co/new) → tipo **Model**.

Ou pela CLI:

```bash
hf repo create edufilgueira/meu-qwen3-8b-merged --type model
```

Para repositório privado:

```bash
hf repo create edufilgueira/meu-qwen3-8b-merged --type model --private
```

---

## 4. Faça o upload da pasta mergeada

Na raiz do `treinamento_llms`:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload edufilgueira/meu-qwen3-8b-merged trein/outputs/merged_model --repo-type model
```

Troque:

- `edufilgueira/meu-qwen3-8b-merged` → nome do seu repo
- `trein/outputs/merged_model` → pasta onde está o merge

Equivalente com a CLI antiga:

```bash
huggingface-cli upload edufilgueira/meu-qwen3-8b-merged trein/outputs/merged_model
```

Para Qwen3-8B mergeado, espere **~16 GB** e um upload que pode levar bastante tempo.

---

## 5. Download para a pasta atual
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download edufilgueira/meu-qwen3-8b-merged
```

## 6. (Opcional) README / model card

Vale a pena um `README.md` no repo com:

- modelo base (`Qwen/Qwen3-8B`)
- dataset usado no treino
- hiperparâmetros (`max_seq_length`, epochs, etc.)
- licença (modelos Qwen têm licença própria — respeite os termos do base)

Você pode subir o README depois:

```bash
hf upload edufilgueira/meu-qwen3-8b-merged README.md --repo-type model
```

---

## Alternativa em Python

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="trein/outputs/merged_model",
    repo_id="edufilgueira/meu-qwen3-8b-merged",
    repo_type="model",
)
```

---

## Depois do upload — testar

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("edufilgueira/meu-qwen3-8b-merged")
model = AutoModelForCausalLM.from_pretrained("edufilgueira/meu-qwen3-8b-merged")
```

---

## Resumo rápido


| Passo      | Comando                                                               |
| ---------- | --------------------------------------------------------------------- |
| Login      | `hf login`                                                            |
| Criar repo | `hf repo create usuario/nome --type model`                            |
| Upload     | `hf upload usuario/nome trein/outputs/merged_model --repo-type model` |


**Importante:** suba a pasta do **merge** (modelo completo), não só o adapter LoRA em `lora_adapter`*. O adapter sozinho não funciona como modelo standalone no Hub da mesma forma.

Se quiser, posso montar um `README.md` modelo para colar no repo do Hugging Face com os campos certos para o seu Qwen3-8B.