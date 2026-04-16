# Organização de datasets para treino (LoRA)

Este documento resume **como organizar** muitas fontes de dados (conversas, livros, Bíblia, psicologia, sintéticos, etc.) de forma sustentável à medida que o volume **cresce**. Complementa o [README.md](README.md) principal, que foca na pipeline técnica (`train_lora.py`).

---

## 1. Por que organizar além de “um JSONL gigante”

Quando você mistura **origens diferentes** (GPT, DeepSeek, curadoria de livros, matriz de dores emocionais, Bíblia, comentários sobre testemunhos, psicologia, etc.):

- o modelo tende a aprender o **estilo da fonte que domina** em quantidade;
- fica difícil **depurar** (“piorou depois que entrou o lote X”) sem saber o que entrou em cada treino;
- **proveniência** (de onde veio cada exemplo) ajuda a rebalancear, remover ou reescrever blocos inteiros.

Por isso: **separar por fonte no armazenamento** costuma ser melhor do que jogar tudo num único arquivo desde o primeiro dia — e **definir explicitamente** o que vira cada **snapshot de treino**.

---

## 2. Estrutura de pastas sugerida

```
data/
  raw/                          # verdade por fonte (não apague o histórico)
    conversas_gpt/              # .jsonl por lote ou data
    conversas_deepseek/
    livros_carta_cristo/
    livros_vida_mestres/
    biblia/
    testemunhos_biblicos/
    psicologia/
    sinteticos_dores_emocionais/
    exemplo/                    # exemplo mínimo (exemplo_treino.jsonl)
  snapshots/                    # gerado por build_snapshot.py
    train_2025-03_v1.jsonl      # sobrescrito enquanto v1 + prefixo fixos
    train_2025-04_v2.jsonl      # exemplo após subir para v2 (novo ficheiro)
```

| Pasta | Função |
|-------|--------|
| **`raw/`** | Um ou mais `.jsonl` por **fonte** (e subpastas). O script `build_snapshot.py` percorre **tudo** recursivamente e unifica. |
| **`snapshots/`** | Ficheiro único `train_<prefixo>_<versão>.jsonl` gerado pelo `build_snapshot.py`. Versão e prefixo vêm de **`data_config.py`**. |

Opcional: um arquivo leve **`data/manifest.csv`** ou **`metadata/`** listando arquivo, domínio, licença, notas — útil quando a equipe cresce.

### 2.1 `data_config.py`, `build_snapshot.py` e versões

Guia dedicado à lógica (escolha de versão, sobrescrita, `--train_file`): **[README_SNAPSHOT.md](README_SNAPSHOT.md)**.

| Ficheiro | Papel |
|----------|--------|
| **`data_config.py`** | Constantes `DATASET_VERSION` (ex.: `"v1"`) e `SNAPSHOT_DATE_PREFIX` (ex.: `"2025-03"`). Enquanto **não** mudarem, o `build_snapshot.py` **sobrescreve** o mesmo ficheiro (`train_2025-03_v1.jsonl`). |
| **`build_snapshot.py`** | Junta todos os `.jsonl` em `data/raw/` num único snapshot em `data/snapshots/`. Embaralha por defeito (`--no-shuffle` para desligar). Cada linha deve ter `messages`; campos extra são removidos na saída. |

Ao subir para **v2** (por exemplo): em `data_config.py` defina `DATASET_VERSION = "v2"` e `SNAPSHOT_DATE_PREFIX = "2025-04"` (ou outro nome de período). O próximo `build_snapshot.py` cria **`train_2025-04_v2.jsonl`** sem apagar o v1.

O **`train_lora.py`**, se **não** passar `--train_file`, escolhe automaticamente o snapshot com **maior número de versão** em `data/snapshots/` (`v2` > `v1`; em empate de versão, ficheiro **maior** em bytes). Se não existir nenhum snapshot, usa `data/exemplo_treino.jsonl` se existir.

**Comandos:**

```bash
python build_snapshot.py
python train_lora.py   # usa o snapshot mais recente por versão
```

---

## 3. Formato das linhas

O treino atual espera **JSONL** com a coluna **`messages`** (lista de `role` / `content`). Veja o [README.md](README.md), Parte B.

Para organização futura, você pode **guardar mais campos** nos arquivos em `raw/` (o pipeline pode ignorar colunas extras se você gerar um snapshot só com `messages`):

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "source": "deepseek",
    "domain": "espiritualidade",
    "collection": "livros_carta_cristo",
    "notes": "opcional"
  }
}
```

Antes de chamar `train_lora.py`, use um script ou processo que produza **`snapshots/train_*.jsonl`** contendo **apenas** o que o loader aceita (por exemplo só `messages`), **ou** evolua o código de carregamento para ler `meta` e fazer *sampling* estratificado — isso é uma extensão opcional do projeto.

---

## 4. Um pool único vs. vários “produtos” de treino

| Estratégia | Quando usar |
|------------|-------------|
| **Um único mix** (várias fontes num snapshot) | Você quer **uma persona única** que una espiritualidade, Bíblia, psicologia e testemunhos no **mesmo estilo** de resposta. Exige **curadoria** e **balanceamento** para um domínio não “esmagar” os outros. |
| **Pools separados** (snapshots diferentes por tema/produto) | Você quer **comportamentos distintos** (ex.: só acolhimento emocional vs. só estudo bíblico). Treinos separados ou **adapters LoRA** separados carregados em contextos diferentes. |
| **Conversas brutas (GPT / DeepSeek)** | Podem ensinar **dois estilos de assistente** ao mesmo tempo. Se a meta é **uma voz só**, considere **filtrar, reescrever ou dar menos peso** a essas fontes em relação ao texto já alinhado ao seu tom (livros curados, etc.). |

---

## 5. Versionar o snapshot de cada treino

Trate cada corrida de treino como um **artefato nomeado**, por exemplo:

- `data/snapshots/train_2025-03_v3.jsonl`

No README interno ou num bloco de notas, registre:

- quais pastas de `raw/` entraram;
- **proporções aproximadas** por domínio (ex.: 30% acolhimento, 20% bíblico);
- o que mudou em relação ao snapshot anterior.

Assim, quando o comportamento do modelo mudar, você sabe **o que mudou no mix**, não só “o dataset cresceu”.

---

## 6. Crescimento contínuo

- **Não apague** o histórico em `raw/` sem backup; prefira **novos arquivos** ou versões datadas.
- Ao **aumentar** um bloco (ex.: mais sintéticos), revise se ele não está **dominando** o próximo snapshot.
- **Deduplicação** e remoção de respostas vazias ou truncadas melhoram a qualidade por megabyte.

---

## 7. Ordem no arquivo e embaralhamento (`shuffle`)

Não há uma regra universal: o que importa é o **tipo de dado** e o **objetivo** do treino. O parâmetro `shuffle_dataset` no `SFTConfig` (TRL) controla se o treino **embaralha** os exemplos ao montar batches — o `train_lora.py` não fixa isso explicitamente; vale conferir o **padrão** da sua versão do TRL ou passar o argumento se quiser controle fino.

### 7.1 O que cada valor significa

| Valor | Efeito |
|-------|--------|
| **`shuffle_dataset=True`** | O treino **embaralha** os exemplos (em geral a cada época). A ordem das linhas no JSONL **não** é seguida de forma fixa. |
| **`shuffle_dataset=False`** | **Não** embaralha: prevalece a **ordem** do dataset (como no arquivo, após preprocessamento). |

Ou seja: quem “sorteia” por você no treino é **`True`**, não `False`. `False` serve quando você **quer ordem fixa** de propósito.

### 7.2 Ordem no arquivo **não** “ensina” separação de temas

**Embaralhar ou não** não é o principal mecanismo para o modelo aprender a **separar** espiritualidade de medicina, etc. O que define domínio e limites é o **conteúdo de cada exemplo** (pergunta, resposta, tom). **Blocos enormes** no arquivo (milhares de linhas só de um tema, depois só de outro) com `shuffle=False` podem até criar **viés** (no início o otimizador vê quase só um tipo) sem isso ser um *currículo* planejado.

### 7.3 Onde o embaralhar **ajuda** (tendência)

- Exemplos **independentes** (cada linha = um diálogo completo): batches mais **mistos** e gradientes mais **representativos** do dataset inteiro.
- **Um único modelo** (ex.: um oráculo) com **vários temas**: reduz o risco de épocas em que os batches ficam **dominados** por um só tipo de caso se o arquivo estava em blocos.
- Caso típico de SFT em chat: **`shuffle=True`** costuma ser o padrão sensato.

### 7.4 Onde **não** embaralhar pode fazer sentido

- **Currículo explícito**: você ordenou o arquivo de propósito (do mais fácil ao mais difícil, ou etapas que só fazem sentido nessa sequência).
- **Depuração**: reproduzir erros com ordem fixa de batches.

### 7.5 Onde **`shuffle=False`** tende a **prejudicar** (sem currículo)

- Arquivo em **blocos longos** por tema **sem** um plano pedagógico: as primeiras atualizações favorecem o que veio **primeiro** no arquivo; não é uma forma confiável de “ensinar compartimentos isolados”.

### 7.6 Fronteira prática (resumo)

| Situação | Tendência |
|----------|-----------|
| Chat SFT, vários temas, exemplos i.i.d. | **`shuffle=True`** em geral **melhora** estabilidade e representatividade. |
| Currículo **desenhado** no arquivo | **`shuffle=False`** pode fazer sentido. |
| Blocos enormes por tema, sem currículo | **`shuffle=False`** costuma **piorar** ou gerar viés de ordem; **`True`** costuma **ajudar**. |
| Dataset pequeno | O gargalo costuma ser **quantidade/qualidade**, não só o shuffle. |

### 7.7 Boas práticas alinhadas a este guia

- **Organize por tema em `raw/`**; ao montar o **snapshot** para treino, **concatene** e **embaralhe** as linhas (ou deixe `shuffle_dataset=True` no treino) — assim você não depende da ordem no disco para “misturar” temas.
- **Separar** espiritualidade de medicina no **comportamento** vem da **curadoria** de cada resposta (e, se necessário, de regras no produto ou adapters separados) — não de desligar o shuffle como atalho.

---

## 8. Cuidados específicos (curadoria)

| Tema | Observação |
|------|------------|
| **Psicologia e dores emocionais** | Reduzir risco de **conselho clínico indevido**: no texto-alvo, tenda a escuta, segurança e, quando couber, **encaminhamento a profissionais**. Volume sozinho não substitui critério de segurança. |
| **Bíblia e comentários** | Se a **versão/tradução** importa para você, mantenha consistência; separe **citação** de **paráfrase** quando fizer sentido. |
| **Livros e obras** | Atenção a **direitos autorais**: treinar com trechos longos pode ter implicação legal; prefira material permitido, trechos curtos, ou **reformulações** suas. |
| **Dados sintéticos** | Úteis para cobrir lacunas; valide uma **amostra** para não amplificar padrões indesejados. |

---

## 9. Resumo prático

1. Manter **vários datasets em `raw/`** por origem/tema, com **metadados** mínimos (`source`, `domain`, etc.).
2. Rodar **`python build_snapshot.py`** para gerar/atualizar o snapshot em `data/snapshots/` (ou ajustar proporções em `raw/` antes de voltar a rodar). Definir **mix** e proporções conscientes — não apenas “concatenar sem critério”.
3. Para um modelo único com vários temas: **embaralhar** o snapshot ou usar **`shuffle_dataset=True`** no treino (ver **§7**); ordem no arquivo sozinha não “separa” domínios — faz isso a **curadoria** de cada exemplo.
4. Se a persona final for **única**, **unificar o tom** na curadoria (e ponderar fontes cruas de chat).
5. Se no futuro houver **modos** claros (ex.: só escuta vs. só estudo), considerar **snapshots ou LoRAs separados** por modo.

---

## 10. Ligação com a pipeline atual

- **[README_SNAPSHOT.md](README_SNAPSHOT.md)** — explica passo a passo a lógica do snapshot, versões e escolha automática no treino.
- **`build_snapshot.py`** — unifica `data/raw/**/*.jsonl` → `data/snapshots/train_<prefixo>_<versão>.jsonl` (ver `data_config.py`).
- **`train_lora.py`** — sem `--train_file`, usa o snapshot de **maior versão** em `data/snapshots/`; com `--train_file caminho.jsonl`, força um ficheiro concreto.
- O [README.md](README.md) descreve instalação, hiperparâmetros, inferência, merge e cache.
