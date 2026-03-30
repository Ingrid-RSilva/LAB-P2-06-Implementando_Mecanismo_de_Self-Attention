# TAREFA 1: O Motor de Frequências
def get_stats(vocab):
    """
    Recebe o dicionário de vocabulário e retorna as frequências
    de todos os pares adjacentes de caracteres/símbolos.
    
    Args:
        vocab (dict): dicionário {sequência_de_chars: frequência}
    
    Returns:
        dict: dicionário {(char1, char2): contagem_total}
    """
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


# TAREFA 2: O Loop de Fusão
def merge_vocab(pair, v_in):
    """
    Recebe o par mais frequente e o vocabulário atual, substituindo
    todas as ocorrências desse par isolado pela versão unificada.
    
    Args:
        pair (tuple): par de símbolos a ser fundido, ex: ('e', 's')
        v_in (dict): vocabulário atual
    
    Returns:
        dict: novo vocabulário com o par fundido
    """
    import re
    v_out = {}
    # Cria o padrão que casa o par com espaços ao redor (ou bordas)
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    for word in v_in:
        new_word = pattern.sub(replacement, word)
        v_out[new_word] = v_in[word]
    return v_out


# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":

    # --- Tarefa 1: Inicialização e validação do Motor de Frequências ---
    print("TAREFA 1: Motor de Frequências")

    vocab = {
        'l o w </w>': 5,
        'l o w e r </w>': 2,
        'n e w e s t </w>': 6,
        'w i d e s t </w>': 3
    }

    stats = get_stats(vocab)

    # Exibe todos os pares ordenados por frequência (decrescente)
    print("\nFrequências de todos os pares adjacentes:")
    for pair, freq in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq}")

    # Validação obrigatória
    assert stats[('e', 's')] == 9, "ERRO: par ('e','s') deveria ter contagem 9!"
    print(f"\n✔ Validação OK: par ('e', 's') = {stats[('e', 's')]} (esperado: 9)")

    # --- Tarefa 2: Loop Principal de Treinamento (5 iterações) ---
    print("TAREFA 2: Loop de Fusão (K=5 iterações)")

    num_merges = 5

    for i in range(1, num_merges + 1):
        stats = get_stats(vocab)
        best_pair = max(stats, key=lambda x: stats[x])
        vocab = merge_vocab(best_pair, vocab)

        print(f"\n--- Iteração {i} ---")
        print(f"  Par fundido: {best_pair}  →  '{' '.join(best_pair)}'")
        print(f"  Vocabulário atual:")
        for word, freq in vocab.items():
            print(f"    '{word}': {freq}")

    print("\n✔ Após 5 iterações é possível observar tokens morfológicos")
    print("  como o sufixo 'est</w>' formado naturalmente pelo BPE.")

    # --- Tarefa 3: Integração Industrial e WordPiece ---
    print("TAREFA 3: WordPiece com BERT Multilingual")

    try:
        from transformers import AutoTokenizer

        print("\nCarregando tokenizador bert-base-multilingual-cased...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
        tokens = tokenizer.tokenize(frase)

        print(f"\nFrase original:\n  {frase}")
        print(f"\nTokens WordPiece ({len(tokens)} tokens):")
        print(f"  {tokens}")

    except ImportError:
        print("\n[AVISO] Biblioteca 'transformers' não instalada.")
        print("Execute: pip install transformers")
    except Exception as e:
        print(f"\n[ERRO] {e}")
