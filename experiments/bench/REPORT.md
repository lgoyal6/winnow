# Winnow compression bench - results

- dataset: `LongBench` | tasks: ['multifieldqa_en', 'hotpotqa'] | examples: 20 | max_context_words: 2500
- correct = token-F1 >= 0.5; scoring = SQuAD-normalized F1/EM (deterministic)

| arm | mean F1 | EM | correct/total | compression | notes |
|---|---|---|---|---|---|
| lingua | 0.337 | 3/20 | 7/20 | 52.5% kept (1.91x) |  |
| union | 0.407 | 6/20 | 8/20 | 62.8% kept (1.59x) |  |
| intersection | 0.251 | 2/20 | 5/20 | 26.8% kept (3.74x) |  |
| vanilla_llm | 0.376 | 4/20 | 8/20 | 1.00x (baseline) | full fp16 KV |
| llm_tq | 0.437 | 6/20 | 9/20 | 3.77x KV | 4.00 eff_bits |
| vanilla_lclm | 0.554 | 8/20 | 10/20 | 1.00x KV / 15.97x seq | 16.00 eff_bits |
| lclm_tq | 0.491 | 7/20 | 9/20 | 3.78x KV / 15.97x seq | 4.00 eff_bits |
