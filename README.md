# Análise de sentimento por aspecto em português com QLoRA

Fine-tuning de um LLM pequeno (Qwen2.5-1.5B-Instruct, 4-bit) para ler uma avaliação de cliente e devolver o sentimento por aspecto em JSON:

```
"Produto ótimo, mas o pedido atrasou 10 dias e ninguém respondeu."
→ {"produto":"positivo","entrega":"negativo","atendimento":"negativo"}
```

Os notebooks rodam no **Google Colab (GPU T4)**. O adaptador publicado tem poucos MB.

- 🤗 Adaptador: [`thiag8ito/qwen2.5-1.5b-aspectos-ecommerce-ptbr-qlora`](https://huggingface.co/thiag8ito/qwen2.5-1.5b-aspectos-ecommerce-ptbr-qlora)
- Notebooks: `01_preparar_dados_colab.ipynb` (dados e rótulos), `02_qlora_avaliacao_publicacao_colab.ipynb` (treino, avaliação, publicação) e `03_analise_de_erros_colab.ipynb` (análise de erros), em `notebooks/`

## Por que esse problema 

Avaliações misturam aspectos: "produto ótimo, entrega péssima" não cabe em uma nota única. Saber **qual** aspecto falhou permite rotear o problema (logística, loja, suporte) e priorizar ações. O dataset usado é o Olist (e-commerce), então os aspectos são `produto`, `entrega` e `atendimento`.

## Pipeline

1. **Dados:** Olist, texto de 25–400 caracteres, deduplicado, amostra balanceada por nota (~3.000).
2. **Rótulos silver:** Qwen2.5-7B-Instruct em 4-bit rodando no Colab, uma avaliação por pedido, guia de anotação fixo com 4 exemplos few-shot e decodificação determinística (sem amostragem).
3. **Teste gold:** 300 avaliações separadas e **revisadas por humano** (`300` de 300 linhas revisadas, ~10% alteradas em relação ao LLM). Concordância LLM × humano (kappa) por aspecto: produto `0.84`, entrega `0.88`, atendimento `0.78`. Indica a qualidade dos rótulos silver usados no treino. Regras de decisão na seção **Guia de anotação**.
4. **Baselines:** (a) o próprio Qwen zero-shot, (b) TF-IDF + Regressão Logística.
5. **QLoRA:** NF4 + LoRA r=16 em todas as projeções, loss só na resposta, 2 épocas.
6. **Avaliação, análise de erros, publicação** com model card gerado a partir das métricas reais.

## Guia de anotação

Cada aspecto é julgado separadamente, **apenas pelo texto** (a nota da avaliação não é usada). Valores: `positivo`, `negativo`, `neutro`, `ausente`.

| Aspecto | Inclui |
|---|---|
| `produto` | qualidade, defeito, funcionamento, tamanho, fidelidade à foto/descrição, item errado ou faltando |
| `entrega` | prazo, atraso, rastreamento, item que não chegou, embalagem danificada no transporte |
| `atendimento` | resposta do vendedor/loja, SAC, troca, devolução, reembolso, comunicação |

| Valor | Quando usar |
|---|---|
| `ausente` | O texto não trata do aspecto. Não se infere o que não foi dito. |
| `neutro` | Há menção sem polaridade clara ou com avaliação equilibrada ("produto ok, nada demais"). |
| `positivo` / `negativo` | Polaridade explícita ou claramente implícita ("não chegou" → `entrega=negativo`). |

**Casos de borda e decisão adotada**

- Produto errado recebido → `produto=negativo`.
- "Loja resolveu rápido" → `atendimento=positivo`.
- "Recomendo a loja" genérico, sem citar atendimento → `atendimento=ausente`.
- Avaliação mista ("produto bom, mas atrasou") → um rótulo por aspecto, não o mesmo para todos.
- Ironia ("Parabéns, só demorou 3 meses") → rotulado pelo sentido real (`entrega=negativo`).

**Processo:** rótulos silver gerados por LLM com este mesmo guia (few-shot, decodificação determinística); depois, revisão manual das linhas do teste, corrigindo o que divergia do guia. Erros recorrentes do LLM observados na revisão: marcar `ausente` quando o aspecto está implícito; confundir `neutro` com `ausente`; ignorar ironia).

## Resultados (conjunto de teste, n=300, rótulo humano)

| Modelo | JSON válido | F1 macro médio | Exact match | F1 menção |
|---|---|---|---|---|
| Qwen2.5-1.5B zero-shot | 0.96 | 0.472 | 0.267 | 0.758 |
| TF-IDF + LogReg | 1.00 | 0.587 | 0.487 | 0.838 |
| **Qwen2.5-1.5B + QLoRA** | **1.00** | **0.691** | **0.733** | **0.923** |

F1 macro e acurácia por aspecto:

| Modelo | F1 produto | F1 entrega | F1 atendimento | Acc produto | Acc entrega | Acc atendimento |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B zero-shot | 0.575 | 0.522 | 0.319 | 0.693 | 0.680 | 0.480 |
| TF-IDF + LogReg | 0.690 | 0.604 | 0.466 | 0.767 | 0.803 | 0.740 |
| **Qwen2.5-1.5B + QLoRA** | **0.805** | **0.689** | **0.577** | **0.877** | **0.913** | **0.887** |

**Leitura:** o QLoRA sobe o F1 macro médio de 0.472 para 0.691 em relação ao zero-shot e supera o TF-IDF em 0.104 pontos de F1 macro e 0.246 de exact match; além disso, a saída passa a ser sempre um JSON válido. `atendimento` é o aspecto mais difícil: a acurácia é alta (0.887), mas o F1 macro é bem menor (0.577), o que sugere classes raras com pouco suporte (hipótese a confirmar no relatório por classe do notebook 03).

![loss](results/loss_curve.png)

**Curva de loss:** a loss de validação cai de 0.074 (passo 50) para 0.051 (passo 300) sem voltar a subir, e a de treino fica um pouco abaixo (cerca de 0.03 a 0.04), sem sinal claro de overfitting em 2 épocas. A loss é medida só nos tokens da resposta, boa parte deles o esqueleto fixo do JSON, então valores baixos não equivalem a acurácia alta; o que mede a qualidade é o F1 no teste.

### Análise de erros

Fonte: `results/erros_modelo.csv` (gerado pelo notebook 03). O modelo errou **97 pares avaliação × aspecto**, em **80 das 300 avaliações** (26,7%; 63 avaliações com um erro e 17 com dois). Isso confere com o exact match de 0.733 (220 de 300 totalmente corretas).

**Erros por tipo e aspecto** (tipos atribuídos automaticamente):

| Tipo de erro | produto | entrega | atendimento | Total |
|---|---|---|---|---|
| Menção perdida (humano marcou o aspecto, modelo previu `ausente`) | 14 | 11 | 24 | **49** (51%) |
| Menção falsa (humano marcou `ausente`, modelo previu sentimento) | 8 | 6 | 6 | 20 (21%) |
| Neutro confundido com polaridade | 12 | 5 | 3 | 20 (21%) |
| Polaridade invertida (positivo ↔ negativo) | 3 | 4 | 1 | 8 (8%) |
| **Total** | **37** | **26** | **34** | **97** |

Trocas mais frequentes (humano → modelo): `negativo→ausente` (22), `ausente→negativo` (14), `neutro→ausente` (14), `positivo→ausente` (13), `neutro→negativo` (7). Por nota, 31 dos 97 erros (32%) estão em avaliações de nota 1.

**O que os erros mostram** (leitura dos 97 casos; as contagens por palavras-chave são aproximadas):

1. **O modelo é conservador: prefere `ausente`.** Metade dos erros (49) são menções perdidas, e em `atendimento` são 24 de 34. Mais da metade dessas menções perdidas de atendimento envolve termos de troca, devolução, contato ou resposta da loja, em frases em que o atendimento está implícito ("... Quero devolver", "como ficamos?"). Em contrapartida, a polaridade invertida é rara (8%): quando o modelo reconhece o aspecto, na grande maioria das vezes acerta o sentimento.

2. **Fronteira entre aspectos em "não recebi" e pedido incompleto** (cerca de 40% dos erros envolvem não recebimento, entrega parcial ou item errado). O mesmo tipo de caso aparece rotulado de formas diferentes: "PRODUTO NÃO RECEBIDO" tem `produto=negativo` no gabarito, enquanto "não recebi o produto no qual já paguei" tem `produto=ausente`; "Comprei 2 calculadoras e só recebi 1" tem `atendimento=negativo`. O modelo também oscila entre `produto`, `entrega` e `atendimento` nesses casos. A causa principal é a ambiguidade do guia de anotação, que ainda não define em qual aspecto cai o item não entregue.

3. **Elogio genérico atribuído a aspectos.** Em frases como "Tudo transcorreu bem e superou minhas expectativas!" ou "Como sempre, compra segura", o gabarito marca `entrega` e `atendimento` como positivos, enquanto o guia manda não inferir (`ausente`). Sete dos 13 casos de menção perdida com `positivo` no gabarito são desse tipo. Aqui o "erro" é discordância entre o gabarito e o próprio guia.

4. **Neutro, concessões e avaliações mistas** (20 erros de neutro confundido, 12 deles em `produto`). Frases com "mas", "porém" ou "embora" aparecem em 8 desses 20 e em metade das polaridades invertidas. Exemplos: "Daria 5 estrelas se a entrega de ambos os produtos fossem juntos! ... Satisfeita" (gabarito `entrega=neutro`, modelo `negativo`) e uma avaliação que critica o valor do frete, mas reconhece que o prazo foi cumprido. O limite entre `neutro` e polaridade é subjetivo e vulnerável a divergências.

5. **Erros reais de leitura.** O modelo de 1,5B erra raciocínios com datas e contrastes. "A entrega estava prevista para 22/08, chegou dia 05/08" recebeu `entrega=negativo` (era positivo), e "entregue 2 dias depois da data prevista ... muito satisfeita com o produto" recebeu `entrega=positivo`. Em "Muito bonita a cortina, só deveria ser melhor o material desfia muito fácil", o elogio dominou a ressalva.

6. **Gabarito possivelmente incorreto.** Alguns erros apontam para o rótulo humano, e não para o modelo: uma avaliação sobre um tapete menor que o anunciado tem `entrega=positivo` no gabarito; "Produto pirata ... solicitei a troca" tem `atendimento=positivo`; uma reclamação de caixa amassada e parafusos soltos tem `atendimento=negativo`; e uma avaliação que elogia apenas a entrega tem `produto=positivo`. Esses casos (ids 2230, 741, 817 e 2323, em `results/erros_modelo.csv`) devem ser revisados, e se corrigidos as métricas medidas sobem um pouco.

**O que fazer a respeito**
- **Guia de anotação:** definir explicitamente onde caem "item não recebido" e "pedido incompleto" (sugestão: `entrega`), elogios genéricos (`ausente`) e pedidos de troca ou devolução sem avaliação do atendimento (`atendimento=neutro`). Depois, **re-rotular e re-revisar** os casos de borda, no treino (silver) e no teste.
- **Menos conservadorismo:** mais exemplos de atendimento implícito e de `neutro` no treino, ou um limiar de decisão para `ausente`, já que o erro dominante é deixar de marcar o aspecto.
- **Modelo e contexto:** testar o Qwen2.5-3B ou um prompt com exemplos de casos de borda para os erros de leitura (datas, contrastes).
- **Medição:** com um segundo anotador sobre uma amostra do teste, medir a concordância humano × humano. Esse é o teto realista do problema.

## Decisões e trade-offs

- **Gerativo em vez de cabeça de classificação:** saída estruturada, fácil de estender para novos aspectos sem mudar a arquitetura. O custo é inferência mais lenta que um BERT; por isso o TF-IDF entra como referência de custo-benefício.
- **Loss mascarada no prompt:** o gradiente só vem da resposta, o que acelera a convergência e evita decorar o prompt.
- **fp16 e não bf16:** a T4 não suporta bf16.
- **Rótulos silver + teste humano:** treinar com anotação barata e avaliar com anotação cara é um desenho comum em produção.
- **Mesmo prompt no baseline e no ajustado:** comparação justa. O ganho vem do fine-tuning, não de um prompt melhor.

## Limitações

Rótulos de treino herdam vieses do LLM anotador; domínio é e-commerce 2016–2018, não delivery; classe `neutro` é rara; sem calibração de confiança.

## Próximos passos

Few-shot e prompt engineering como baseline mais forte; ablação de `r` (8/16/32) e de 0.5B vs 1.5B; calibração e abstenção com baixa confiança; servir com vLLM/merge do adaptador e medir latência e custo por mil avaliações; monitoramento de drift.

## Como rodar

O projeto tem duas partes: **notebook 1 local** (VS Code, só CPU, usa a API da Anthropic) e **notebook 2 no Colab** (fine-tuning na GPU T4).

### Parte 1: notebook 1, no VS Code

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # edite e preencha ANTHROPIC_API_KEY
```

1. Abra a pasta no VS Code (extensões *Python* e *Jupyter*) e escolha o kernel `.venv` no notebook.
2. Rode `notebooks/01_preparar_dados.ipynb`. Comece com `N_LABEL = 100` para testar e ver o consumo de tokens.
3. Revise `data/test_para_revisao.csv` (Excel, Sheets ou LibreOffice) e salve como `data/test_revisado.csv`.
4. Rode as células finais: concordância e geração do `data.zip`.

**Custo da API:** com 10 avaliações por chamada, ~3.000 avaliações usam da ordem de 0,3 milhão de tokens de entrada e 0,1 milhão de saída (estimativa; o notebook imprime o uso real).

**Alternativa sem a API:** `notebooks/01_preparar_dados_colab.ipynb` faz toda a preparação no Colab, rotulando com o Qwen2.5-7B-Instruct na GPU (grátis, mais lento e com rótulos um pouco mais ruidosos). Ele gera o mesmo `data.zip`.

### Parte 2: notebook 2, no Colab

1. Em colab.research.google.com, *Arquivo → Fazer upload de notebook* e envie `notebooks/02_qlora_avaliacao_publicacao_colab.ipynb`.
2. *Ambiente de execução → Alterar tipo → T4 GPU*.
3. Rode a primeira célula e selecione o `data.zip`.
4. Ajuste `HF_USER` e faça login no Hugging Face (`notebook_login`) quando chegar na publicação.
5. No fim, baixe o `resultados.zip` (métricas, curva de loss e model card) antes de encerrar a sessão.

Se a GPU gratuita estiver bloqueada, espere liberar ou use o Pay As You Go: uma execução completa do notebook 2 deve consumir poucas unidades.

### Parte 3: análise de erros (Colab)

Rode `notebooks/03_analise_de_erros_colab.ipynb` com o `data.zip`. Ele carrega o adaptador do Hugging Face, confere as métricas, gera relatório por classe, matrizes de confusão, contagem de erros por tipo e o `results/erros_modelo.csv` para categorização manual.

Versões usadas: 

accelerate==1.15.0  
bitsandbytes==0.50.2  
peft==0.21.2  
sentence-transformers==5.7.0  
torch==2.11.0+cu130  
torchao==0.10.0  
torchaudio==2.11.0+cu130  
torchcodec==0.11.0+cu130  
torchdata==0.11.0  
torchsummary==1.5.1  
torchtune==0.6.1  
torchvision==0.26.0+cu130  
transformers==5.22.0  


