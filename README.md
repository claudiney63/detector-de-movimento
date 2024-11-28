# Detecção de Movimento com Fluxo Óptico
Este repositório contém as implementações das atividades práticas relacionadas à disciplina de Visão Computacional. As atividades exploram diferentes abordagens para a detecção de movimento em sequências de imagens. A seguir, são descritas as duas partes do trabalho.

## Parte 1: Detecção de Movimento por Histogramas

Nesta etapa, foi desenvolvido um sistema que:
  1 - Analisa o histograma de quadros consecutivos em uma sequência de cenas.
  2 - Calcula as diferenças entre os histogramas para identificar regiões onde ocorreu movimento.
  3 - Gera uma saída visual, destacando as áreas em que mudanças significativas foram detectadas.
  
### Arquitetura
  - Entrada: Sequência de quadros ou frames de vídeo.
  - Processamento:
    - Extração dos histogramas (usando OpenCV).
    - Cálculo da diferença entre histogramas para identificar alterações.
  - Saída: Imagem destacando as regiões com movimento e sinalizando um "alarme visual".

## Parte 2: Detecção de Movimento com Fluxo Óptico
Nesta segunda etapa, o objetivo foi implementar um sistema mais avançado, baseado no cálculo do fluxo óptico, para detectar e segmentar objetos em movimento.

### Descrição
O sistema:
  1 - Estima o fluxo óptico entre pares de frames, utilizando métodos como Lucas-Kanade ou Horn-Schunck.
  2 - Segmenta os objetos em movimento com base nos dados de magnitude e direção do fluxo.
  3 - Gera uma imagem segmentada que ilustra os objetos detectados.
  
### Arquitetura
  - Entrada: Sequência de quadros ou frames de vídeo.
  - Processamento:
    - Cálculo do fluxo óptico para estimar o movimento.
    - Segmentação baseada na magnitude do movimento.
  - Saída: Imagem segmentada, ilustrando objetos distintos em movimento.

