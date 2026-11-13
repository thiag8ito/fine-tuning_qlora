# Análise de erros do adaptador QLoRA

← [Voltar ao README](../README.md)

Análise dos erros do modelo `Qwen2.5-1.5B + QLoRA` no conjunto de teste (300 avaliações, rótulo humano).

- **Fonte dos dados:** [`results/erros_modelo.csv`](../results/erros_modelo.csv), gerado pelo notebook `03_analise_de_erros_colab.ipynb`.
- **Tipos de erro** atribuídos automaticamente pelo notebook; a **leitura qualitativa** (itens 1 a 5) foi feita sobre os 97 casos, e as contagens por palavras-chave são aproximadas.

## Visão geral

O modelo errou **97 pares avaliação × aspecto**, em **80 das 300 avaliações** (26,7%; 63 avaliações com um erro e 17 com dois). Isso confere com o exact match de 0.733 (220 de 300 totalmente corretas).

| Tipo de erro | produto | entrega | atendimento | Total |
|---|---|---|---|---|
| Menção perdida (humano marcou o aspecto, modelo previu `ausente`) | 14 | 11 | 24 | **49** (51%) |
| Menção falsa (humano marcou `ausente`, modelo previu sentimento) | 8 | 6 | 6 | 20 (21%) |
| Neutro confundido com polaridade | 12 | 5 | 3 | 20 (21%) |
| Polaridade invertida (positivo ↔ negativo) | 3 | 4 | 1 | 8 (8%) |
| **Total** | **37** | **26** | **34** | **97** |

Trocas mais frequentes (humano → modelo): `negativo→ausente` (22), `ausente→negativo` (14), `neutro→ausente` (14), `positivo→ausente` (13), `neutro→negativo` (7). Por nota, 31 dos 97 erros (32%) estão em avaliações de nota 1.

## O que os erros mostram

1. **O modelo é conservador: prefere `ausente`.** Metade dos erros (49) são menções perdidas, e em `atendimento` são 24 de 34. Mais da metade dessas menções perdidas de atendimento envolve termos de troca, devolução, contato ou resposta da loja, em frases em que o atendimento está implícito ("... Quero devolver", "como ficamos?"). Em contrapartida, a polaridade invertida é rara (8%): quando o modelo reconhece o aspecto, na grande maioria das vezes acerta o sentimento.

2. **Fronteira entre aspectos em "não recebi" e pedido incompleto** (cerca de 40% dos erros envolvem não recebimento, entrega parcial ou item errado). O mesmo tipo de caso aparece rotulado de formas diferentes: "PRODUTO NÃO RECEBIDO" tem `produto=negativo` no gabarito, enquanto "não recebi o produto no qual já paguei" tem `produto=ausente`; "Comprei 2 calculadoras e só recebi 1" tem `atendimento=negativo`. O modelo também oscila entre `produto`, `entrega` e `atendimento` nesses casos. A causa principal é a ambiguidade do guia de anotação, que ainda não define em qual aspecto cai o item não entregue.

3. **Elogio genérico atribuído a aspectos.** Em frases como "Tudo transcorreu bem e superou minhas expectativas!" ou "Como sempre, compra segura", o gabarito marca `entrega` e `atendimento` como positivos, enquanto o guia manda não inferir (`ausente`). Sete dos 13 casos de menção perdida com `positivo` no gabarito são desse tipo. Aqui o "erro" é discordância entre o gabarito e o próprio guia.

4. **Neutro, concessões e avaliações mistas** (20 erros de neutro confundido, 12 deles em `produto`). Frases com "mas", "porém" ou "embora" aparecem em 8 desses 20 e em metade das polaridades invertidas. Exemplos: "Daria 5 estrelas se a entrega de ambos os produtos fossem juntos! ... Satisfeita" (gabarito `entrega=neutro`, modelo `negativo`) e uma avaliação que critica o valor do frete, mas reconhece que o prazo foi cumprido. O limite entre `neutro` e polaridade é subjetivo e vulnerável a divergências.

5. **Erros reais de leitura.** O modelo de 1,5B erra raciocínios com datas e contrastes. "A entrega estava prevista para 22/08, chegou dia 05/08" recebeu `entrega=negativo` (era positivo), e "entregue 2 dias depois da data prevista ... muito satisfeita com o produto" recebeu `entrega=positivo`. Em "Muito bonita a cortina, só deveria ser melhor o material desfia muito fácil", o elogio dominou a ressalva.

## Impacto nas métricas

Os itens 2 e 3 indicam que parte dos 97 erros vem de **ruído e inconsistência do gabarito**, e não de falha do modelo. Por isso o F1 medido deve ser lido como um limite inferior do desempenho real, e o teto do problema é definido pela concordância entre anotadores humanos, que não foi medida.

## O que fazer a respeito

- **Guia de anotação:** definir explicitamente onde caem "item não recebido" e "pedido incompleto" (sugestão: `entrega`), elogios genéricos (`positivo`) e pedidos de troca ou devolução sem avaliação do atendimento (`atendimento=neutro`). Depois, **re-rotular e re-revisar** os casos de borda, no treino (silver) e no teste.
- **Menos conservadorismo:** mais exemplos de atendimento implícito e de `neutro` no treino, ou um limiar de decisão para `ausente`, já que o erro dominante é deixar de marcar o aspecto.
- **Modelo e contexto:** testar o Qwen2.5-3B ou um prompt com exemplos de casos de borda para os erros de leitura (datas, contrastes).
- **Medição:** com um segundo anotador sobre uma amostra do teste, medir a concordância humano × humano. Esse é o teto realista do problema.
