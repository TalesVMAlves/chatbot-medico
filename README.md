# Chatbot Médico Inteligente

Este projeto tem como objetivo desenvolver um agente inteligente para auxiliar no diagnóstico de doenças respiratórias, como Gripe, Resfriado, Alergia e COVID-19, com base nos sintomas apresentados pelos usuários. Utilizando técnicas de Recuperação Aumentada por Geração (RAG) e análise de dados preditiva, o sistema busca fornecer as informações baseado em evidên.

## 📝 Visão Geral do Projeto

O sistema se propõe a:

*   **Fornecer Informações Confiáveis:** Utilizar a base de conhecimento oficial do Ministério da Saúde (SUS) para gerar respostas informativas e precisas sobre as doenças em questão.
*   **Calcular Probabilidades de Sintomas:** Implementar modelos de machine learning para estimar a probabilidade de um usuário ter determinada doença com base nos sintomas reportados.
*   **Acessibilidade e Confiabilidade:** Basear as análises em dados públicos e amplamente reconhecidos, garantindo a qualidade e a confiabilidade das informações.

## 📚 Base de Conhecimento e Fontes de Dados

### Documentos de Orientações do SUS (para RAG)

A base de conhecimento para o RAG foi construída a partir de documentos oficiais do Ministério da Saúde, garantindo a atualização e a confiabilidade das informações sobre Gripe (Influenza) e COVID-19, bem como diretrizes gerais de saúde.

*   **Gripe (Influenza):**
    *   [Saúde de A a Z: Gripe Influenza](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/g/gripe-influenza)
    *   [Guia de Manejo e Tratamento de Influenza 2023](https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/influenza/guia-de-manejo-e-tratamento-de-influenza-2023)
*   **COVID-19:**
    *   [Saúde de A a Z: COVID-19](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/c/covid-19)
    **Resfriado:**
    **Alergia Resíratória:**
*   **Diretrizes Gerais de Saúde:**
    *   [Protocolos Clínicos e Diretrizes Terapêuticas (PCDT)](https://www.gov.br/saude/pt-br/assuntos/pcdt)

### Dataset de Sintomas (para Análise Preditiva)

Para o treinamento e avaliação de modelos de classificação de sintomas, utilizamos um dataset público disponível no Kaggle, que abrange sintomas de Gripe, Resfriado, Alergia e COVID-19.

*   **Sintomas de Gripe, Resfriado, Alergia e COVID-19:**
    *   [Link do Dataset no Kaggle](https://www.kaggle.com/datasets/walterconway/covid-flu-cold-symptoms)

### SRAG (Síndrome Respiratória Aguda Grave) - DATASUS

O link para o SRAG do DATASUS esteve indisponível no momento da criação deste README, mas é reconhecido como uma fonte de dados extremamente relevante para análises aprofundadas sobre doenças respiratórias.

*   **Trabalho de Referência sobre Dados SRAG:** Recomendamos a consulta ao seguinte trabalho, que utiliza dados semelhantes:
    *   [URL do Artigo na MDPI](https://www.mdpi.com/2076-3417/13/20/11518)

## 🚀 Tecnologias e Abordagens

*   **Recuperação Aumentada por Geração (RAG):** Implementação de um sistema RAG para consulta e sumarização de informações dos documentos do SUS.
*   **Machine Learning:** Utilização de modelos preditivos para classificar doenças com base nos sintomas.
*   **PyCaret:** Ferramenta de AutoML para agilizar o processo de construção e avaliação de modelos de machine learning.

## 🧪 Experimentos e Análises Recomendadas

Incentivamos a exploração e a replicação de análises utilizando as fontes de dados disponibilizadas:

1.  **Análise com PyCaret e Dataset do Kaggle:**
    *   Realizar a importação e o pré-processamento do dataset de sintomas do Kaggle.
    *   Utilizar o PyCaret para experimentar diferentes modelos de classificação (ex: Logistic Regression, RandomForestClassifier, GradientBoostingClassifier) para prever a doença com base nos sintomas.
    *   Avaliar a performance dos modelos utilizando métricas apropriadas (acurácia, precisão, recall, F1-score).
2.  **Análise com Dados SRAG e PyCaret (quando disponíveis):**
    *   Após a disponibilidade do link do DATASUS para SRAG, explorar a possibilidade de coletar e pré-processar esses dados.
    *   Aplicar as mesmas técnicas do PyCaret para construir modelos preditivos, buscando insights sobre os padrões de SRAG.
