---
language: pt
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags: [peft, qlora, lora, portuguese, aspect-based-sentiment-analysis, olist, brazil]
datasets: [olistbr/brazilian-ecommerce]
---

# thiag8ito/qwen2.5-1.5b-aspectos-ecommerce-ptbr-qlora

Adaptador **QLoRA** (LoRA r=16 sobre base 4-bit NF4) de `Qwen/Qwen2.5-1.5B-Instruct` para **análise de sentimento por aspecto** em avaliações de e-commerce em português do Brasil. Dada uma avaliação, retorna JSON com o sentimento de `produto`, `entrega` e `atendimento`.

## Uso

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, "thiag8ito/qwen2.5-1.5b-aspectos-ecommerce-ptbr-qlora")

SYSTEM = "Você classifica avaliações de clientes de e-commerce brasileiro por aspecto.\nAspectos: produto (qualidade/estado do item), entrega (prazo/transporte), atendimento (vendedor/suporte/troca).\nValores: positivo, negativo, neutro, ausente (aspecto não mencionado).\nResponda somente com JSON: {\"produto\":\"...\",\"entrega\":\"...\",\"atendimento\":\"...\"}"
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Produto ótimo, mas o pedido atrasou 10 dias e ninguém respondeu."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=40, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

O prompt de sistema deve ser **o mesmo usado no treino** (acima).

## Dados

- Fonte: avaliações do [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (texto de avaliações reais, 25–400 caracteres, deduplicadas, amostra balanceada por nota).
- Rótulos de treino/validação: **silver**, gerados por um LLM maior (Qwen/Qwen2.5-7B-Instruct (4-bit, few-shot)) com guia de anotação fixo.
- Teste: 300 avaliações **revisadas manualmente** (gold).
- Split: 2400 treino / 300 validação / 300 teste.

## Treinamento

| Item | Valor |
|---|---|
| Quantização | 4-bit NF4, double quant, compute fp16 |
| LoRA | r=16, alpha=32, dropout=0.05, todas as projeções lineares |
| Otimizador | paged AdamW 8-bit, lr 2e-4 cosine, warmup 3% |
| Batch efetivo | 16 (8 x 2 acumulação) |
| Épocas | 2 |
| Loss | apenas nos tokens da resposta |
| Hardware | GPU T4 (Colab gratuito) |

## Avaliação (conjunto de teste, rótulo humano)

| modelo                 |   json_valido |   f1_produto |   acc_produto |   f1_entrega |   acc_entrega |   f1_atendimento |   acc_atendimento |   f1_macro_medio |   exact_match |   f1_mencao |
|:-----------------------|--------------:|-------------:|--------------:|-------------:|--------------:|-----------------:|------------------:|-----------------:|--------------:|------------:|
| Qwen2.5-1.5B zero-shot |          0.96 |        0.575 |         0.693 |        0.522 |         0.68  |            0.319 |             0.48  |            0.472 |         0.267 |       0.758 |
| TF-IDF + LogReg        |          1    |        0.69  |         0.767 |        0.604 |         0.803 |            0.466 |             0.74  |            0.587 |         0.487 |       0.838 |
| Qwen2.5-1.5B + QLoRA   |          1    |        0.805 |         0.877 |        0.689 |         0.913 |            0.577 |             0.887 |            0.691 |         0.733 |       0.923 |

`f1_*`: F1 macro por aspecto. `exact_match`: três aspectos corretos. `f1_mencao`: detecção de menção do aspecto. Saídas inválidas contam como erro.

## Limitações e riscos

- Domínio: e-commerce (Olist, 2016–2018). **Não foi validado em delivery de comida**; para esse uso, ajuste com dados do domínio.
- Rótulos de treino são silver: o modelo herda os vieses e erros do LLM anotador.
- Ironia, sarcasmo e avaliações mistas são pontos fracos conhecidos.
- Classe `neutro` é rara; as métricas nela têm alta variância.
- Avaliações podem conter dados pessoais não removidos; o dataset original é anonimizado pelo Olist.
- Modelo de apoio à análise; não usar para decisões automáticas sobre pessoas ou lojistas sem revisão humana.

## Reprodução

Código e notebooks: https://github.com/thiag8ito/fine-tunning_qlora

