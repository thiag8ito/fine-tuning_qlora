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

**Processo:** rótulos silver gerados por LLM com este mesmo guia (few-shot, decodificação determinística); depois, revisão manual das linhas do teste, corrigindo o que divergia do guia. Erros recorrentes do LLM observados na revisão: marcar `ausente` quando o aspecto está implícito, confundir `neutro` com `ausente`, ignorar ironia.

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

**Leitura:** o QLoRA sobe o F1 macro médio de 0.472 para 0.691 em relação ao zero-shot e supera o TF-IDF em 0.104 pontos de F1 macro e 0.246 de exact match; além disso, a saída passa a ser sempre um JSON válido. `atendimento` é o aspecto mais difícil: a acurácia é alta (0.887), mas o F1 macro é bem menor (0.577), o que sugere classes raras com pouco suporte.

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
- **Guia de anotação:** definir explicitamente onde caem "item não recebido" e "pedido incompleto" (sugestão: `entrega`), elogios genéricos (`positivo`) e pedidos de troca ou devolução sem avaliação do atendimento (`atendimento=neutro`). Depois, **re-rotular e re-revisar** os casos de borda, no treino (silver) e no teste.
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

Todo o projeto roda no **Google Colab** com GPU T4 (gratuita ou com unidades de computação), sem GPU local e **sem chave de API**. Os rótulos são gerados por um modelo aberto (Qwen2.5-7B-Instruct) dentro do próprio Colab. Os notebooks são executados em ordem e trocam dados pelo arquivo `data.zip`.

**Pré-requisitos:** conta Google (Colab) e conta no Hugging Face com um token de **escrita** (huggingface.co/settings/tokens), necessário só para publicar o adaptador no notebook 2.

| # | Notebook | O que faz | Entrada | Saída | Tempo na T4 |
|---|---|---|---|---|---|
| 1 | [`01_preparar_dados_colab`](notebooks/01_preparar_dados_colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thiag8ito/fine-tunning_qlora/blob/main/notebooks/01_preparar_dados_colab.ipynb) | Baixa o Olist, rotula com o Qwen 7B, separa treino/validação/teste, exporta o teste para revisão e calcula o kappa | CSV revisado | `data.zip` | ~30 min |
| 2 | [`02_qlora_avaliacao_publicacao_colab`](notebooks/02_qlora_avaliacao_publicacao_colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thiag8ito/fine-tunning_qlora/blob/main/notebooks/02_qlora_avaliacao_publicacao_colab.ipynb) | Mede o baseline, treina o QLoRA, avalia e publica o adaptador | `data.zip` | `resultados.zip` e adaptador no Hugging Face | ~1 h |
| 3 | [`03_analise_de_erros_colab`](notebooks/03_analise_de_erros_colab.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thiag8ito/fine-tunning_qlora/blob/main/notebooks/03_analise_de_erros_colab.ipynb) | Carrega o adaptador do Hub e analisa os erros no teste | `data.zip` | `erros_modelo.csv` | ~5 min |

### Passo a passo

1. **Notebook 1.** Abra pelo botão, escolha *Ambiente de execução → Alterar tipo → T4 GPU* e rode as células em ordem. Ao fim da rotulagem, o `test_para_revisao.csv` é baixado automaticamente.
2. **Revisão humana.** Abra o CSV no Google Sheets, corrija `produto`, `entrega` e `atendimento` conforme o [guia de anotação](#guia-de-anotação), baixe como CSV e envie na célula seguinte. Ela confere se o arquivo pertence ao mesmo split, calcula o kappa e verifica que treino, validação e teste não se sobrepõem. Por fim, baixe o `data.zip`.
3. **Notebook 2.** Em uma sessão nova com T4, envie o `data.zip` na primeira célula, ajuste `HF_USER` e faça o login quando a célula de publicação pedir o token. Antes de treinar, confira a linha `passos totais` (~300 com 2.400 amostras de treino). No fim, baixe o `resultados.zip`.
4. **Notebook 3.** Em outra sessão, envie o `data.zip` de novo. Ele gera `results/erros_modelo.csv`, que alimenta a análise de erros.

Os arquivos do Colab somem quando a sessão termina, então baixe o `data.zip`, o `resultados.zip` e o `erros_modelo.csv` antes de fechar.

### Reprodutibilidade

- As sementes são fixas (`SEED = 42`), e as versões testadas estão em [Versões usadas](#versões-usadas).
- A rotulagem silver e o treino em 4-bit na GPU não são totalmente determinísticos, então reexecutar pode dar números ligeiramente diferentes.
- O teste revisado **não é distribuído** (o Olist tem licença própria). Quem repetir o projeto gera o seu `test_para_revisao.csv` e faz a própria revisão, e as métricas podem diferir.
- Se a GPU gratuita estiver bloqueada, espere liberar ou use o Pay As You Go do Colab.  

Versões usadas: 

python=3.12.10
accelerate==1.15.0  
bitsandbytes==0.50.2  
peft==0.21.2  
torch==2.11.0+cu130  
torchao==0.10.0  
torchcodec==0.11.0+cu130  
torchdata==0.11.0  
torchsummary==1.5.1    
torchvision==0.26.0+cu130  
transformers==5.22.0  


