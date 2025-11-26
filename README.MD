# Projeto de Análise de Clustering para E-commerce

## Descrição do Projeto

Este projeto realiza uma análise completa de clustering em dados de e-commerce utilizando técnicas de aprendizado de máquina não supervisionado. A análise é dividida em duas abordagens principais: segmentação de clientes usando K-Means e detecção de anomalias em produtos usando DBSCAN.

## Objetivos

### 1. Segmentação de Clientes (K-Means)
- Identificar grupos de clientes com comportamentos de compra similares
- Desenvolver perfis de clientes baseados em características RFM (Recência, Frequência, Valor Monetário)
- Fornecer insights para estratégias de marketing direcionadas

### 2. Detecção de Anomalias (DBSCAN)
- Identificar produtos com padrões de vendas atípicos
- Detectar possíveis problemas no catálogo de produtos
- Fornecer alertas para revisão de preços e desempenho de produtos

## Estrutura do Projeto

```
projeto-individual-machine-learning-2/
│
├── data/                    # Arquivos CSV originais
│   ├── customers.csv
│   ├── products.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── suppliers.csv
│   ├── reviews.csv
│   ├── payment.csv
│   └── shipments.csv
│
├── src/                    # Código fonte
│   ├── data_loader.py     # Carregamento de dados
│   ├── preprocessor.py    # Pré-processamento e feature engineering
│   ├── kmeans_analyzer.py # Análise K-Means
│   ├── dbscan_analyzer.py # Análise DBSCAN
│   ├── visualizer.py      # Geração de visualizações
│   └── main.py           # Script principal
│
└── results/               # Resultados da análise
    ├── plots/            # Gráficos e visualizações
    └── reports/          # Relatórios em CSV
```

## Metodologia

### Pré-processamento de Dados

#### Features para Clientes (K-Means)
- **Recência**: Dias desde a última compra
- **Frequência**: Número total de pedidos
- **Valor Monetário**: Valor total gasto pelo cliente
- **Rating Médio**: Avaliação média dada pelo cliente

#### Features para Produtos (DBSCAN)
- **Preço**: Preço do produto
- **Quantidade Total**: Volume total de vendas
- **Contagem de Pedidos**: Número de pedidos que incluem o produto
- **Rating Médio**: Avaliação média do produto

### Algoritmos Utilizados

#### K-Means Clustering
- Técnica de clustering particional
- Número ótimo de clusters determinado pelo método do cotovelo
- Métrica de avaliação: Silhouette Score

#### DBSCAN (Density-Based Spatial Clustering)
- Detecção de outliers baseada em densidade
- Não requer número pré-definido de clusters
- Identifica automaticamente pontos anômalos

## Resultados Esperados

### K-Means - Segmentação de Clientes
- **Cluster 0**: Clientes Inativos - Alta recência, baixa frequência e valor
- **Cluster 1**: Clientes Regulares - Recência e frequência moderadas
- **Cluster 2**: Clientes Premium - Baixa recência, alta frequência e valor
- **Cluster 3**: Clientes Valiosos - Altíssimo valor, bons ratings

### DBSCAN - Detecção de Anomalias
- **Produtos com preço muito alto**: Possíveis erros de cadastro ou produtos premium
- **Produtos com baixo volume**: Itens com pouca saída que precisam de revisão
- **Produtos com rating baixo**: Itens com problemas de qualidade ou descrição

## Como Executar

### Pré-requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execução
```bash
cd src
python main.py
```

### Arquivos de Entrada Necessários
Coloque os seguintes arquivos CSV na pasta `data/`:
- customers.csv
- products.csv
- orders.csv
- order_items.csv
- reviews.csv
- payment.csv (opcional)
- shipments.csv (opcional)
- suppliers.csv (opcional)

## Saídas Geradas

### Relatórios (results/reports/)
- `analysis_summary.csv`: Resumo estatístico da análise
- `customer_segmentation.csv`: Dados segmentados dos clientes
- `product_anomalies.csv`: Produtos identificados como anomalias

### Visualizações (results/plots/)
- `kmeans_elbow_method.png`: Método do cotovelo para K-Means
- `kmeans_clustering_results.png`: Resultados da segmentação de clientes
- `dbscan_anomaly_detection.png`: Detecção de anomalias em produtos
- `methods_comparison.png`: Comparação entre os métodos

## Interpretação dos Resultados

### Segmentação de Clientes
A análise de clusters permite:
- Desenvolver campanhas de marketing personalizadas
- Criar programas de fidelidade segmentados
- Otimizar estratégias de retenção de clientes
- Identificar clientes de alto valor para atendimento premium

### Detecção de Anomalias
A identificação de outliers ajuda a:
- Revisar estratégias de preços
- Identificar produtos problemáticos
- Otimizar o mix de produtos
- Melhorar a qualidade do catálogo

## Aplicações Práticas

### Para a Equipe de Marketing
- Desenvolvimento de campanhas específicas por segmento
- Estratégias de retenção para clientes inativos
- Programas de fidelidade para clientes premium

### Para a Gestão de Produtos
- Revisão de produtos com preços anômalos
- Análise de produtos com baixo desempenho
- Otimização do catálogo baseada nos clusters de desempenho

### Para a Tomada de Decisão
- Alocação de recursos baseada em segmentos de clientes
- Desenvolvimento de estratégias de preços
- Planejamento de estoque e logística

## Considerações Técnicas

### Limitações do K-Means
- Requer definição prévia do número de clusters
- Sensível a outliers
- Assume clusters esféricos e de tamanho similar

### Limitações do DBSCAN
- Sensível aos parâmetros eps e min_samples
- Dificuldade com clusters de densidades variadas
- Pode ter problemas com dados de alta dimensionalidade

### Próximos Passos Sugeridos
1. Implementar campanhas segmentadas baseadas nos clusters
2. Revisar estratégia de preços para produtos anômalos
3. Desenvolver sistema de alerta para novos padrões anômalos
4. Integrar análise com dados em tempo real
5. Expandir análise com features adicionais

## Conclusão

Este projeto demonstra a aplicação prática de técnicas de clustering não supervisionado para extrair insights valiosos de dados de e-commerce. A combinação de K-Means para segmentação e DBSCAN para detecção de anomalias fornece uma visão abrangente do comportamento de clientes e desempenho de produtos, permitindo decisões baseadas em dados para otimização de negócios.

Os resultados obtidos podem ser utilizados para melhorar estratégias de marketing, otimizar o catálogo de produtos e aumentar a eficiência operacional do e-commerce.