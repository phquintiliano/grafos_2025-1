# Alunos: Pedro Henrique Santana Quintiliano 202120142; Raul Souza Lima 202010709

# 📊 Trabalho de Grafos – Etapa 1: Pré-processamento de Dados

Este projeto foi desenvolvido como parte da disciplina de **Teoria dos Grafos**, com foco na **Etapa 1**, que envolve o **pré-processamento dos dados**. O objetivo principal é representar o problema em estruturas de dados de grafos, realizar a leitura e estruturação do arquivo `.dat` e calcular métricas estatísticas com base nos caminhos mínimos entre todos os pares de vértices.

---

## 🚀 Objetivos da Etapa 1

- Representar a modelagem do problema utilizando **estruturas de dados em grafos**;
- Implementar a **leitura de arquivos de entrada** do tipo `.dat`;
- Calcular **métricas estatísticas** baseadas na matriz de caminhos mínimos gerada pelo **algoritmo de Floyd-Warshall**.

---

## 📦 Estrutura do Projeto

### Funções principais

- `read_file(arquivo)` → Lê e organiza os dados do arquivo `.dat`;
- `metrics(...)` → Calcula todas as métricas necessárias para o grafo;
- `find_components()` → Encontra componentes conectados com BFS;
- `find_degree()` → Retorna grau mínimo e máximo dos vértices;
- `find_intermediation()` → Calcula a intermediação com base na matriz de predecessores;
- `find_medium_weight()` → Calcula o caminho médio e o diâmetro do grafo.

---

## 📈 Métricas Calculadas

1. Quantidade de vértices;
2. Quantidade de arestas;
3. Quantidade de arcos;
4. Quantidade de vértices requeridos;
5. Quantidade de arestas requeridas;
6. Quantidade de arcos requeridos;
7. **Densidade do grafo** (_order strength_);
8. Componentes conectados;
9. Grau mínimo dos vértices;
10. Grau máximo dos vértices;
11. **Intermediação** – frequência com que um nó aparece nos caminhos mínimos;
12. **Caminho médio** entre os vértices;
13. **Diâmetro** – o maior caminho mínimo do grafo.

---

## 🧠 Algoritmo Utilizado

O projeto utiliza o **algoritmo de Floyd-Warshall** para:

- Calcular a **matriz de distâncias mínimas** entre todos os pares de vértices;
- Calcular a **matriz de predecessores**, utilizada para métricas como intermediação.

---

## 💻 Execução

1. Siga os passos do Grafos1.ipynb
