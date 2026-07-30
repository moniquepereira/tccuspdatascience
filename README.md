Informações + scripts resultantes do trabalho de conclusão de curso de MBA USP/ESALQ em Data Science & Analytics de Monique Madeira Pereira (Turma 242)

Scripts:
01_scraping_dados*.py - procedimento para coleta de dados do site da FIVB para competições entre seleções nacionais femininas e masculinas; resultam nos arquivos com dados brutos por partida/set/atleta
02_tratamento_dados*.py - procedimento para transformação dos dados do site da FIVB - cálculo de dados sobre eficiência, tratamento de ausências, etc.; resultam nos arquivos com dados tratados (cada set representado por uma linha no dataset, separando competições masculinas e femininas)
03_algoritmos_classificacao.py - treinamento e teste dos modelos de machine learning.

Diretórios:
/dados_brutos: dados coletados a partir dos scripts 01_scraping_dados.py e 01_scraping_dados_masculino.py
/dados_tratados: dados coletados a partir dos scripts 02_tratamento_dados.py e 02_tratamento_dados_masculino.py / entradas de dados para o script 03_algoritmos_classificacao.py
/imagens: imagens de matriz de confusão e curva ROC geradas pelo script 03_algoritmos_classificacao.py
/modelos: arquivos pickle dos modelos treinados para reprodução nos dados de seleções masculinas - gerados e reutilizados pelo script 03_algoritmos_classificacao.py
/resultados: arquivos csv com as comparações de resultados entre algoritmos e estatísticas de contribuição de variáveis - geradas pelo script 03_algoritmos_classificacao.py
