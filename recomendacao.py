import pandas as pd
import numpy as np
import sys

portifolioNome = sys.argv[1]

# Parâmetros

QTDE_VIZINHOS = 20
#TAM_AMOSTRA = 100
TAM_AMOSTRA_PERCENTUAL = 0.6



try:
    df = pd.read_csv('../src/Mercado_Formatado_Final.csv', sep=',', decimal='.', encoding = "UTF-8")
    df = df.drop(['Unnamed: 0'], axis=1)
except:
    print("Base de dados nao encontrada")
    print("Verifique se o mesmo se encontra em:")
    print("../src/Mercado_Formatado_Final.csv")
    sys.exit()

try:
    p1 = pd.read_csv("../src/" + portifolioNome)
    p1 = p1.drop(['Unnamed: 0'], axis=1)
except:
    print("Problemas na leitura do arquivo")
    print("Verifique se o mesmo se encontra em:")
    print("../src/" + portifolioNome)
    sys.exit()

dfSemID = df.drop('id',axis=1)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=QTDE_VIZINHOS, n_jobs=-1 )

model_knn.fit(dfSemID)

"""### Funções auxiliareas"""

# Transforma uma lista de índices em uma lista de IDs

def returnID (df, list_indices):
  list_ids = []

  for i in list_indices:
    list_ids.append(df.iloc[i]['id'])

  return list_ids

# Retorna os vizinhos para cada empresa de uma lista de empresas

def recomenda(lista_empresas, qtde_vizinhos):

  temp = {"Origem":[],"Vizinho":[],"Distância":[]}

  for empresa in lista_empresas:

    vizinhos = model_knn.kneighbors(dfSemID.iloc[[empresa]])

    a = [empresa] * qtde_vizinhos
    b = list(vizinhos[1][0])
    c = list(vizinhos[0][0])

    a_ids = returnID(df, a)
    b_ids = returnID(df, b)

    temp['Origem'].extend(a_ids)
    temp['Vizinho'].extend(b_ids)
    temp['Distância'].extend(c)

  # Monta o dataframe e apaga as linhas em que a origem e vizinho sao iguais
  result = pd.DataFrame(data=temp)
  result = result[result['Origem'] != result['Vizinho']]

  return result

def amostra_portfolio(df_mercado, df_portfolio, tam_amostra):

  # Amostra o portfolio, retornando uma lista de IDs
  amostra_IDs = df_portfolio.sample(n = tam_amostra)['id'].tolist()

  # Pega os índices (na Mercado) dos IDs amostrados
  amostra_indices = df_mercado.index[df_mercado['id'].isin(amostra_IDs)]

  return amostra_indices

# Flag para recomendações fora da amostra
# Adiciona uma coluna no dataframe de recomendações, informando se é Diferente de alguém que já existia na amostra

def flag_recom_diferentes(df_recomendacoes):

  vizinhos = set(list(df_recomendacoes['Vizinho']))
  origens = set(list(df_recomendacoes['Origem']))

  diferentes = list(vizinhos - origens)

  df_recomendacoes['Diferente'] = [1 if i in(diferentes) else 0 for i in list(df_recomendacoes['Vizinho'])]

  return df_recomendacoes

# Flag para acerto
# Adiciona uma coluna no dataframe de recomendações, informando se está no Portfolio

def flag_recom_acerto(df_recomendacoes, df_portfolio):

  vizinhos = set(list(df_recomendacoes['Vizinho']))
  portfolio = set(list(df_portfolio['id']))

  acertos = vizinhos.intersection(portfolio)

  df_recomendacoes['Acerto'] = [1 if i in(acertos) else 0 for i in list(df_recomendacoes['Vizinho'])]

  return df_recomendacoes

# Junta a flag para recomendações fora da amostra e flag para acerto

def flag_recom_diferentes_e_acerto(df_recomendacoes):

  df_recomendacoes['Diferente e Acerto'] = df_recomendacoes['Acerto'] & df_recomendacoes['Diferente']

  return df_recomendacoes

# Calcula medidas de cada rodada
#   - Quantidade de recomendações no total
#   - Quantidade de recomendações únicas
#   - Quantidade de recomendações fora da amostra
#   - Quantidade de recomendações fora da amostra únicas
#   - Quantidade de acertos
#   - Quantidade de acertos únicos
#   - Quantidade de recomendações fora da amostra e acertos
#   - Quantidade de recomendações fora da amostra e acertos únicas

def calcula_medidas_rodada(df_recomendacoes):

  qtde_recom_total = len(df_recomendacoes['Vizinho'])
  qtde_recom_unicas = len(df_recomendacoes['Vizinho'].unique())
  qtde_recom_diferentes = df_recomendacoes['Diferente'].sum()
  qtde_recom_diferentes_unicas = len(df_recomendacoes[df_recomendacoes['Diferente'] == 1]['Vizinho'].unique())
  qtde_acertos = df_recomendacoes['Acerto'].sum()
  qtde_acertos_unicos = len(df_recomendacoes[df_recomendacoes['Acerto'] == 1]['Vizinho'].unique())
  qtde_recom_diferentes_e_acertos =  df_recomendacoes['Diferente e Acerto'].sum()
  qtde_recom_diferentes_e_acertos_unicas = len(df_recomendacoes[df_recomendacoes['Diferente e Acerto'] == 1]['Vizinho'].unique())

  return [qtde_recom_total, qtde_recom_unicas,
          qtde_recom_diferentes, qtde_recom_diferentes_unicas,
          qtde_acertos, qtde_acertos_unicos,
          qtde_recom_diferentes_e_acertos, qtde_recom_diferentes_e_acertos_unicas]

p1 = p1.filter(['id'])

TAM_AMOSTRA = round(1.0 * len(p1)) #100%

# Para cada rodada

# Amostra o portfolio
amostra_p1_indices = amostra_portfolio(df, p1, TAM_AMOSTRA)

# Roda o KNN para aquela amostra
recomendacoes_p1 = recomenda(amostra_p1_indices, QTDE_VIZINHOS)

# Flag para recomendações fora da amostra
recomendacoes_p1 = flag_recom_diferentes(recomendacoes_p1)

# Flag para acerto
recomendacoes_p1 = flag_recom_acerto(recomendacoes_p1, p1)

# Junta a flag para recomendações fora da amostra e flag para acerto
recomendacoes_p1 = flag_recom_diferentes_e_acerto(recomendacoes_p1)

leads = list(recomendacoes_p1[(recomendacoes_p1.Diferente == 1) & (recomendacoes_p1.Acerto == 0)]['Vizinho'])
recomendacoes = {}
for i in leads:
  recomendacoes[i] = leads.count(i)

leads = pd.DataFrame(columns=["Lead","Count"])
leads['Lead']= recomendacoes.keys()
leads['count']= recomendacoes.values()
leads = leads.sort_values(by=['Count'], ascending=False)

leads = leads[leads["count"]>=2]['Lead']
leads.to_csv("../src/answer.csv", index=False)
