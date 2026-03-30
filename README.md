# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece

## Descrição

Implementação do motor básico do algoritmo **Byte Pair Encoding (BPE)** do zero e exploração do tokenizador **WordPiece** via Hugging Face, conforme especificado no Laboratório 6 da disciplina.

---

## Estrutura do Projeto

```
├── lab6_bpe_tokenizer.py      # Script Python completo (todas as tarefas)
├── lab6_bpe_tokenizer.ipynb   # Jupyter Notebook com saídas e análises
└── README.md
```

---

## Como executar

```bash
pip install transformers
python lab6_bpe_tokenizer.py
```

---

## Tarefas implementadas

### Tarefa 1 — Motor de Frequências (`get_stats`)

A função `get_stats(vocab)` percorre cada entrada do vocabulário, divide a string em símbolos e conta quantas vezes cada par adjacente aparece, ponderado pela frequência da palavra no corpus.

**Validação:** o par `('e', 's')` retorna contagem **9** (6 de *newest* + 3 de *widest*). ✔

### Tarefa 2 — Loop de Fusão (`merge_vocab` + loop K=5)

A função `merge_vocab(pair, v_in)` substitui, em cada entrada do vocabulário, todas as ocorrências **isoladas** do par pelo token fundido. O loop executa 5 iterações, sempre fundindo o par de maior frequência. Resultado observável após as 5 iterações: formação do token morfológico `est</w>`. ✔

### Tarefa 3 — WordPiece com BERT Multilingual

Tokenização da frase de teste com `bert-base-multilingual-cased`:

```
Entrada: "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

Saída: ['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform',
        '##er', 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente',
        'di', '##f', '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

---

## O que significam os tokens com `##`?

No WordPiece, o prefixo `##` indica que aquele fragmento **não ocorre no início de uma palavra** — ele é a continuação de um token anterior. Por exemplo, `inconstitucionalmente` é decomposto em `in` + `##cons` + `##tit` + `##uc` + `##ional` + `##mente`: o `in` é o início reconhecido pelo vocabulário, e cada `##` sinaliza que o pedaço pertence à mesma palavra original.

Essa estratégia resolve o problema *out-of-vocabulary* (OOV): mesmo que uma palavra nunca tenha aparecido durante o treinamento, ela pode ser representada como uma sequência de sub-palavras conhecidas. O modelo nunca fica sem uma representação numérica válida, pois no pior caso recorre a caracteres individuais presentes no vocabulário base. Isso é fundamental para modelos multilíngues que precisam lidar com morfologia rica e palavras novas em dezenas de idiomas.

---

Partes deste laboratório foram desenvolvidas com auxílio da IA generativa **Claude (Anthropic)**. Especificamente:

- **Tarefa 2 — `merge_vocab`:** a expressão regular utilizada para identificar e substituir pares isolados de tokens (`(?<!\S)bigram(?!\S)`) foi construída com sugestão da IA e revisada pela autora Ingrid para garantir o comportamento correto nas bordas de string.

Todos os demais trechos (estrutura do algoritmo BPE, lógica de contagem de pares, loop principal e integração com Hugging Face) foram escritos e compreendidos pela autora Ingrid.

---

## Versão

`v1.0` — versão final para avaliação.
