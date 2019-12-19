## Lead-recomendation

Dado o conjunto de clientes de uma empresa(portfólio), como remendar novos potenciais clientes(Leads) para ela?

Este trabalho apresenta um 

Modelo de recomendação de Leads Baseado em KNN

O notebook "TratamentoDeFeatures" contem abordagem utilizada para tratar as features da base de dados fornecida pela Codenation. A partir dele gera-se uma nova base de dados em que é aplicado o modelo.

O notebook "aplciacaoModeloTestes" apresenta a aplicação do modelo para os tres portfólios fornecidos para teste
E subsequentes análises

O script "recomendacao.py" é o próprio recomendador que lê a base de dados formatada no script de tratamento de features
lê um arquivo csv contendo o portfólio da empresa e gera um arquivos chamado "answer.csv" contendo a recomendacao de empresas para aquele portfólio.

# Como Executar

python recomendacao.py "NomeArquivoPortfólio".csv
O arquivo de portfólio deverá estar dentro da pasta src bem como a base de dados.

Os pricipais notebooks para a realizacao deste projeto também se encontram no colab

#Tratamento das features
https://colab.research.google.com/drive/1m02cqbr8q0UcBJVBlcHA2e44sc1d6SnH

#Aplicacao Modelo e testes
https://colab.research.google.com/drive/196cvSbCYIKsbi8Ks8OiUbPIl0vnBFwxL

#Base de dados gerada para o modelo
https://drive.google.com/open?id=1bVm9mtHznTY6HG3wmgJZHkSamfM-7_kn
