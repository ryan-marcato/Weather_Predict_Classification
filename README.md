# Weather_Predict_Classification
- Inclui vários recursos relacionados ao clima e categoriza o clima em quatro tipos: Rainy, Sunny, Cloudy, e Snowy.
- Dataset retirado do Kaggle link: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification

# Objetivo
- Fazer uma breve analise grafica buscando informações.
- E realizar previsões da classificação.

# Instalação e Execução do Projeto
Para usar este projeto, você precisa de Python e das seguintes bibliotecas:
1. Instalar Bibliotecas Necessárias
```bash
pip install pandas numpy seaborn matplotlib scikit-plot feature-engine scikit-learn
```

2. Executar o Projeto
Se você estiver trabalhando em um Jupyter Notebook, inicie o Jupyter Notebook com o seguinte comando:
```bash
 jupyter notebook
```
Abra o notebook relevante e execute as células para preparar os dados e treinar os modelos.

3. Preparar os Dados e Modelos
- Importe as Bibliotecas
 No seu notebook, comece importando as bibliotecas necessárias:
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

from feature_engine import encoding
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
```
- Preparar os Dados
Utilize as funções ```train_test_split ``` para dividir seus dados em conjuntos de treino e teste e ```LabelEncoder``` para codificar variáveis categóricas
- Criar e Treinar Modelos
Use ```RandomForestClassifier``` e ```KNeighborsClassifier``` para treinar modelos. Você também pode usar ```GridSearchCV``` para otimizar os hiperparâmetros.
- Avaliar o Modelo
Utilize ```bash metrics``` e ```scikit-plot``` para avaliar o desempenho dos modelos.
