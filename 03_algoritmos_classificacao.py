# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 10:24:51 2026

@author: nique
"""

import pandas as pd
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pickle

#%% Funções

def gerar_matriz_confusao_agregada(tecnica_analitica, variaveis_treino, target_treino, variaveis_teste, target_teste):
    global comparativo_algoritmos
    
    # Calculando a matriz de confusão para treino e teste
    matriz_confusao_treino = confusion_matrix(target_treino, variaveis_treino)
    matriz_confusao_teste = confusion_matrix(target_teste, variaveis_teste)

    # Criando uma figura para as duas matrizes de confusão
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
    
    # Exibindo a matriz de confusão para treino
    matriz_confusao_display_treino = ConfusionMatrixDisplay(matriz_confusao_treino)
    matriz_confusao_display_treino.plot(ax=axs[0], colorbar=False, cmap='Blues')
    axs[0].set_title(tecnica_analitica + ': Base de Treino')
    axs[0].set_ylabel('Observado (Real)')
    axs[0].set_xlabel('Classificado (Modelo)')
    
    # Exibindo a matriz de confusão para teste
    matriz_confusao_display_teste = ConfusionMatrixDisplay(matriz_confusao_teste)
    matriz_confusao_display_teste.plot(ax=axs[1], colorbar=False, cmap='Blues')
    axs[1].set_title(tecnica_analitica + ': Base de Teste')
    axs[1].set_ylabel('Observado (Real)')
    axs[1].set_xlabel('Classificado (Modelo)')

    plt.tight_layout()
    
    nome_grafico = 'matriz_confusao_'+tecnica_analitica+'_feminino.jpg'
    plt.savefig('imagens/'+nome_grafico, format='jpg', dpi=300)
    
    plt.show()   

    # Calculando métricas para treino
    acuracia_treino = accuracy_score(target_treino, variaveis_treino)
    sensibilidade_treino = recall_score(target_treino, variaveis_treino, pos_label=1)
    especificidade_treino = recall_score(target_treino, variaveis_treino, pos_label=0)
    precisao_treino = precision_score(target_treino, variaveis_treino)

    # Calculando métricas para teste
    acuracia_teste = accuracy_score(target_teste, variaveis_teste)
    sensibilidade_teste = recall_score(target_teste, variaveis_teste, pos_label=1)
    especificidade_teste = recall_score(target_teste, variaveis_teste, pos_label=0)
    precisao_teste = precision_score(target_teste, variaveis_teste)

    # Contando VP, FP, VN e FN
    VP_treino = matriz_confusao_treino[1, 1]
    FP_treino = matriz_confusao_treino[0, 1]
    VN_treino = matriz_confusao_treino[0, 0]
    FN_treino = matriz_confusao_treino[1, 0]

    VP_teste = matriz_confusao_teste[1, 1]
    FP_teste = matriz_confusao_teste[0, 1]
    VN_teste = matriz_confusao_teste[0, 0]
    FN_teste = matriz_confusao_teste[1, 0]
 
    comparativo_algoritmos.loc[len(comparativo_algoritmos)] = ({'tecnica_analitica': tecnica_analitica,
                                                                'tipo_base': 'Treino',
                                                                'acuracia': acuracia_treino,
                                                                'sensibilidade': sensibilidade_treino,
                                                                'especificidade': especificidade_treino,
                                                                'precisao': precisao_treino,
                                                                'VP': VP_treino,
                                                                'FP': FP_treino,
                                                                'VN': VN_treino,
                                                                'FN': FN_treino})
    
    comparativo_algoritmos.loc[len(comparativo_algoritmos)] = ({'tecnica_analitica': tecnica_analitica,
                                                                'tipo_base': 'Teste',
                                                                'acuracia': acuracia_teste,
                                                                'sensibilidade': sensibilidade_teste,
                                                                'especificidade': especificidade_teste,
                                                                'precisao': precisao_teste,
                                                                'VP': VP_teste,
                                                                'FP': FP_teste,
                                                                'VN': VN_teste,
                                                                'FN': FN_teste})

    return

def gerar_matriz_confusao(tecnica_analitica, tipo_base, variaveis, target):
    global comparativo_algoritmos
    
    matriz_confusao         = confusion_matrix(variaveis, target)
    matriz_confusao_display = ConfusionMatrixDisplay(matriz_confusao)

    plt.rcParams['figure.dpi'] = 600
    matriz_confusao_display.plot(colorbar=False, cmap='Blues')
    plt.title(''+tecnica_analitica+': Base de '+tipo_base)
    plt.ylabel('Observado (Real)')
    plt.xlabel('Classificado (Modelo)')
    
    plt.tight_layout()
    
    nome_grafico = 'matriz_confusao_'+tecnica_analitica+'_masculino.jpg'
    plt.savefig('imagens/'+nome_grafico, format='jpg', dpi=300)
    
    plt.show()
    
    acuracia       = accuracy_score(target, variaveis)
    sensibilidade  = recall_score(target, variaveis, pos_label=1)
    especificidade = recall_score(target, variaveis, pos_label=0)
    precisao       = precision_score(target, variaveis)
    
    # Contando VP, FP, VN e FN
    VP = matriz_confusao[1, 1]
    FP = matriz_confusao[0, 1]
    VN = matriz_confusao[0, 0]
    FN = matriz_confusao[1, 0]
    
    comparativo_algoritmos.loc[len(comparativo_algoritmos)] = ({'tecnica_analitica': tecnica_analitica,
                                                                'tipo_base': tipo_base,
                                                                'acuracia': acuracia,
                                                                'sensibilidade': sensibilidade,
                                                                'especificidade': especificidade,
                                                                'precisao': precisao,
                                                                'VP': VP,
                                                                'FP': FP,
                                                                'VN': VN,
                                                                'FN': FN})
    return

def gerar_curva_roc_comparacao(tecnica_analitica, origem_dados, tipo_base, variaveis_prob_treino, target_treino, variaveis_prob_teste, target_teste):
    # Parametrizando a curva ROC para o conjunto de treino
    fpr_treino, tpr_treino, _ = roc_curve(target_treino, variaveis_prob_treino[:, 1])
    roc_auc_treino = auc(fpr_treino, tpr_treino)

    # Parametrizando a curva ROC para o conjunto de teste
    fpr_teste, tpr_teste, _ = roc_curve(target_teste, variaveis_prob_teste[:, 1])
    roc_auc_teste = auc(fpr_teste, tpr_teste)

    # Plotando as curvas ROC
    plt.figure(figsize=(15, 10), dpi=600)
    plt.plot(fpr_treino, tpr_treino, color='red', linestyle='dashed', linewidth=3, label='Treino (AUC = %0.3f)' % roc_auc_treino)
    plt.plot(fpr_teste, tpr_teste, color='blue', linewidth=3, label=''+tipo_base+' (AUC = %0.3f)' % roc_auc_teste)
    
    # Configurações do gráfico
    plt.title('Curvas ROC ' + tecnica_analitica + ' (' + origem_dados + ')', fontsize=22)
    plt.xlabel('1 - Especificidade', fontsize=20)
    plt.ylabel('Sensibilidade', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    
    nome_grafico = 'curvas_roc_'+tecnica_analitica+'_'+origem_dados+'.jpg'
    plt.savefig('imagens/'+nome_grafico, format='jpg', dpi=300)
    
    plt.show()
    

    return

def agregar_importancias(lista_dfs_importancia: list, feature_col: str = 'features', importance_col: str = 'importance'):
    """
    Agrega múltiplos dataframes de feature importance, normaliza e calcula a média.

    Args:
        lista_dfs_importancia (list): Uma lista de tuplas no formato [('nome_modelo1', df1), ('nome_modelo2', df2), ...].
        feature_col (str): Nome da coluna que contém os nomes das features.
        importance_col (str): Nome da coluna que contém os valores de importância.

    Returns:
        pd.DataFrame: Um DataFrame com a importância média e normalizada de cada feature.
    """
    # Renomeia a coluna de importância em cada df para ser única
    for nome, df in lista_dfs_importancia:
        df.rename(columns={importance_col: f'importance_{nome}'}, inplace=True)

    # Extrai apenas os DataFrames para o merge
    dfs_para_merge = [df for nome, df in lista_dfs_importancia]

    # Junta todos os DataFrames usando a coluna de features
    df_agregado = reduce(lambda left, right: pd.merge(left, right, on=feature_col, how='outer'), dfs_para_merge)
    df_agregado.fillna(0, inplace=True)

    # Identifica as colunas de importância
    cols_importance = [col for col in df_agregado.columns if col.startswith('importance_')]

    # Normaliza cada coluna de importância para somar 1
    for col in cols_importance:
        total_importance = df_agregado[col].sum()
        if total_importance > 0:
            df_agregado[col] = df_agregado[col] / total_importance

    # Calcula a média das importâncias normalizadas
    df_agregado['importance_media'] = df_agregado[cols_importance].mean(axis=1)
    df_agregado.sort_values(by='importance_media', ascending=False, inplace=True)
    df_agregado.reset_index(drop=True, inplace=True)
    
    # Prepara o DataFrame final
    df_final = df_agregado[[feature_col, 'importance_media']].copy()
    df_final.sort_values(by='importance_media', ascending=False, inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_agregado

#%% Carregando os dados e tratando tipos e dados faltantes
INPUT = "df_consolidado_saida.csv"

df_dados_tratados = pd.read_csv(INPUT, encoding='utf-8')
df_dados_tratados.info()

df_dados_tratados = pd.get_dummies(df_dados_tratados, 
                       columns=['ID_SET'], 
                       drop_first=True,
                       dtype='category')

for col in df_dados_tratados.select_dtypes(include='bool').columns:
    df_dados_tratados[col] = df_dados_tratados[col].astype('category')
    
df_dados_tratados.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
df_dados_tratados = df_dados_tratados.drop(columns=['VENCEU_SET_A_ANTERIOR','VENCEU_SET_B_ANTERIOR'])

# removendo as partidas que possuem algum atributo nan
df_dados_tratados.dropna(how="any", inplace=True)
df_dados_tratados.info()

#%% Definindo a lista de features
variaveis = list(df_dados_tratados.columns)

variaveis.remove('ID_PARTIDA')
variaveis.remove('TIMEA')
variaveis.remove('TIMEB')
variaveis.remove('VENCEU_SET_A')
variaveis.remove('VENCEU_SET_B')

vTarget = 'VENCEU_SET_A'

print(variaveis)
print(vTarget)

#%% Análise descritiva das variáveis
estatisticas_quanti = df_dados_tratados[df_dados_tratados.select_dtypes(include='float64').columns].describe()
print(estatisticas_quanti)

for col in df_dados_tratados.select_dtypes(include='category').columns:
    print('Estatística descritiva ('+col+')')
    print(df_dados_tratados[col].value_counts())
    
#%% Separando as amostras de treino e teste
X = df_dados_tratados[variaveis]
y = df_dados_tratados[vTarget]

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=123)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())

#%% Comparativos entre modelos

comparativo_algoritmos = pd.DataFrame(columns=['tecnica_analitica', 
                                               'tipo_base', 
                                               'acuracia', 
                                               'sensibilidade', 
                                               'especificidade', 
                                               'precisao',
                                               'VP',
                                               'FP',
                                               'VN',
                                               'FN'])

#%%######################### Árvore de Decisão ################################
###############################################################################
#%% Árvore de decisão com alguns hiperparâmetros

tree_clf_aj = DecisionTreeClassifier(max_depth=5,
                                     max_features=3,
                                     min_samples_split=10,
                                     min_samples_leaf=5,
                                     random_state=123)
tree_clf_aj.fit(X_train, y_train)

# Predict na base de treinamento
tree_pred_train_class_aj = tree_clf_aj.predict(X_train)
tree_pred_train_prob_aj = tree_clf_aj.predict_proba(X_train)

# Predict na base de teste
tree_pred_test_class_aj = tree_clf_aj.predict(X_test)
tree_pred_test_prob_aj = tree_clf_aj.predict_proba(X_test)

gerar_matriz_confusao_agregada('Árvore de decisão', tree_pred_train_class_aj, y_train, tree_pred_test_class_aj, y_test)

# Importância das variáveis preditoras
tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':np.round(tree_clf_aj.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('Árvore de Decisão', 'Feminino', 'Teste', tree_pred_train_prob_aj, y_train, tree_pred_test_prob_aj, y_test)

# Salvar modelo pickle
with open('modelos/arvore_decisao.pkl', 'wb') as f:
    pickle.dump(tree_clf_aj, f)

#%%######################### Random Forest (Grid Search) ######################
###############################################################################
#%%

# Vamos especificar a lista de hiperparâmetros desejados e seus valores
param_grid_rf = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 7],
    'max_features': [3, 4],
    'min_samples_split': [20, 50],
    'min_samples_leaf': [5, 10]
}

# Identificar o algoritmo em uso
rf_grid = RandomForestClassifier(random_state=123)

# Treinar os modelos para o grid search
rf_grid_model = GridSearchCV(estimator = rf_grid, 
                             param_grid = param_grid_rf,
                             scoring='accuracy',
                             cv=5, verbose=0)

rf_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
rf_best = rf_grid_model.best_estimator_

# Predict na base de treino
rf_grid_pred_train_class = rf_best.predict(X_train)
rf_grid_pred_train_prob = rf_best.predict_proba(X_train)

# Predict na base de testes
rf_grid_pred_test_class = rf_best.predict(X_test)
rf_grid_pred_test_prob = rf_best.predict_proba(X_test)

# Matriz de confusão 
gerar_matriz_confusao_agregada('Random Forest (Grid Search)', rf_grid_pred_train_class, y_train, rf_grid_pred_test_class, y_test)

# Importância das variáveis preditoras
rf_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':np.round(rf_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('Random Forest (Grid Search)', 'Feminino', 'Teste', rf_grid_pred_train_prob, y_train, rf_grid_pred_test_prob, y_test)

# Salvar modelo pickle
with open('modelos/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_best, f)

#%%######################### XGBoost com Grid Search ##########################
###############################################################################
#%%

# Lista de hiperparâmetros para o Grid Search
param_grid_xgb = {
    'n_estimators': [100,300],
    'max_depth': [3, 5],
    'colsample_bytree': [0.5, 1],
    'learning_rate': [0.01, 0.1]
}

xgb_grid = XGBClassifier(random_state=123, enable_categorical=True)

# Treinar os modelos para o grid search
xgb_grid_model = GridSearchCV(estimator = xgb_grid, 
                              param_grid = param_grid_xgb,
                              scoring='accuracy', 
                              cv=5, verbose=0)

xgb_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
xgb_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
xgb_best = xgb_grid_model.best_estimator_

# Valores preditos na base de treinamento
xgb_grid_pred_train_class = xgb_best.predict(X_train)
xgb_grid_pred_train_prob = xgb_best.predict_proba(X_train)

# Valores preditos na base de teste
xgb_grid_pred_test_class = xgb_best.predict(X_test)
xgb_grid_pred_test_prob = xgb_best.predict_proba(X_test)

# Matriz de confusão 
gerar_matriz_confusao_agregada('XGBoost (Grid Search)', xgb_grid_pred_train_class, y_train, xgb_grid_pred_test_class, y_test)

# Importância das variáveis preditoras
xgb_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(xgb_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('XGBoost (Grid Search)', 'Feminino', 'Teste', xgb_grid_pred_train_prob, y_train, xgb_grid_pred_test_prob, y_test)

# Salvar modelo pickle
with open('modelos/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_best, f)

#%%######################### SVM com Grid Search ##############################
###############################################################################
#%%

# Lista de hiperparâmetros para Grid Search
param_grid = [
  {'C': [0.1, 1, 5], 'kernel': ['linear']},
  {'C': [0.1, 1, 5], 'degree': [2, 3], 'coef0': [0, 1], 'kernel': ['poly']},
  {'C': [0.1, 1, 5], 'gamma': [0.1, 1, 10, 100], 'kernel': ['rbf']},
]

# Identificar o algoritmo em uso
svm_grid = SVC(random_state=123, probability=True)

# Treinar os modelos para o grid search
model_grid = GridSearchCV(estimator = svm_grid, 
                          param_grid = param_grid,
                          scoring='accuracy',
                          cv=5,
                          verbose=0)

model_grid.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
model_grid.best_params_

# Gerando o modelo com os melhores hiperparâmetros
svm_best = model_grid.best_estimator_

# Valores preditos nas bases de treino e teste
svm_grid_train = svm_best.predict(X_train)
svm_grid_train_proba = svm_best.predict_proba(X_train)
svm_grid_test = svm_best.predict(X_test)
svm_grid_test_proba = svm_best.predict_proba(X_test)

# Matriz de confusão 
gerar_matriz_confusao_agregada('SVM (Grid Search)', svm_grid_train, y_train, svm_grid_test, y_test)

# Importância das variáveis preditoras no modelo
perm_svm = permutation_importance(svm_best, X_test, y_test, 
                                  n_repeats=10, 
                                  scoring='accuracy',
                                  random_state=123)

svm_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(perm_svm.importances_mean, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('SVM (Grid Search)', 'Feminino', 'Teste', svm_grid_train_proba, y_train, svm_grid_test_proba, y_test)

# Salvar modelo pickle
with open('modelos/svm.pkl', 'wb') as f:
    pickle.dump(svm_best, f)

#%%######################### K-Nearest Neighbors ##############################
###############################################################################
#%%
knn_pipe = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
knn_pipe.fit(X_train, y_train)

knn_pred_train = knn_pipe.predict(X_train)
knn_pred_train_proba = knn_pipe.predict_proba(X_train)
knn_pred_test = knn_pipe.predict(X_test)
knn_pred_test_proba = knn_pipe.predict_proba(X_test)

# Matriz de confusão 
gerar_matriz_confusao_agregada('K-Nearest Neighbors', knn_pred_train, y_train, knn_pred_test, y_test)

# Importância das variáveis preditoras no modelo
perm_knn = permutation_importance(knn_pipe, X_test, y_test, n_repeats=30, random_state=123, n_jobs=-1)
knn_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(perm_knn.importances_mean, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('K-Nearest Neighbors', 'Feminino', 'Teste', knn_pred_train_proba, y_train, knn_pred_test_proba, y_test)

# Salvar modelo pickle
with open('modelos/knn.pkl', 'wb') as f:
    pickle.dump(knn_pipe, f)

#%%######################### Redes Neurais MLP ################################
###############################################################################
#%%
mlp_pipe = make_pipeline(StandardScaler(), 
                         MLPClassifier(hidden_layer_sizes=(100,), 
                                       activation='logistic',
                                       max_iter=1000, 
                                       random_state=123))
mlp_pipe.fit(X_train, y_train)

# Valores preditos nas bases de treino e teste
mlp_pred_grid_train = mlp_pipe.predict(X_train)
mlp_pred_grid_train_proba = mlp_pipe.predict_proba(X_train)
mlp_pred_grid_test = mlp_pipe.predict(X_test)
mlp_pred_grid_test_proba = mlp_pipe.predict_proba(X_test)

# Matriz de confusão
gerar_matriz_confusao_agregada('Rede Neural MLP', mlp_pred_grid_train, y_train, mlp_pred_grid_test, y_test)

# Análise das variáveis relevantes
perm_mlp = permutation_importance(mlp_pipe, 
                                  X_test, 
                                  y_test, 
                                  n_repeats=30, 
                                  random_state=123, 
                                  n_jobs=-1)

mlp_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(perm_mlp.importances_mean, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Curva ROC (base de teste)
gerar_curva_roc_comparacao('Rede Neural MLP', 'Feminino', 'Teste', mlp_pred_grid_train_proba, y_train, mlp_pred_grid_test_proba, y_test)

# Salvar modelo pickle
with open('modelos/rede_neural_mlp.pkl', 'wb') as f:
    pickle.dump(mlp_pipe, f)

#%% Sumarização dos resultados para base de dados de jogos FEMININOS

lista_de_importancias = [
    ('tree', tree_features),
    ('rf', rf_features),
    ('xgb', xgb_features), 
    ('svm', svm_features),
    ('knn', knn_features),
    ('mlp', mlp_features)
]

lista_importancia_variaveis = agregar_importancias(lista_de_importancias)
lista_importancia_variaveis.to_csv('resultados/lista_importancia_variaveis.csv', index=False, encoding="utf-8")
comparativo_algoritmos.to_csv('resultados/estatisticas_algoritmos.csv', index=False, encoding="utf-8")

#%% APLICAÇÃO DE ALGORITMOS NA BASE DE DADOS DE JOGOS MASCULINOS (E COMPARAÇÃO COM FEMININO)

INPUT = "df_consolidado_saida_masc.csv"

df_dados_tratados_masc = pd.read_csv(INPUT, encoding='utf-8')
df_dados_tratados_masc.info()

df_dados_tratados_masc = pd.get_dummies(df_dados_tratados_masc, 
                                        columns=['ID_SET'], 
                                        drop_first=True,
                                        dtype='category')

for col in df_dados_tratados_masc.select_dtypes(include='bool').columns:
    df_dados_tratados_masc[col] = df_dados_tratados_masc[col].astype('category')
    
df_dados_tratados_masc.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
df_dados_tratados_masc = df_dados_tratados_masc.drop(columns=['VENCEU_SET_A_ANTERIOR','VENCEU_SET_B_ANTERIOR'])

# removendo as partidas que possuem algum atributo nan
df_dados_tratados_masc.dropna(how="any", inplace=True)
df_dados_tratados_masc.info()

#%% Análise descritiva das variáveis
estatisticas_quanti = df_dados_tratados_masc[df_dados_tratados_masc.select_dtypes(include='float64').columns].describe()
print(estatisticas_quanti)

for col in df_dados_tratados_masc.select_dtypes(include='category').columns:
    print('Estatística descritiva ('+col+')')
    print(df_dados_tratados_masc[col].value_counts())
    
#%% Definindo a lista de features
variaveis = list(df_dados_tratados_masc.columns)

variaveis.remove('ID_PARTIDA')
variaveis.remove('TIMEA')
variaveis.remove('TIMEB')
variaveis.remove('VENCEU_SET_A')
variaveis.remove('VENCEU_SET_B')

vTarget = 'VENCEU_SET_A'

print(variaveis)
print(vTarget)
#%% Todas as amostras são para previsão
X_predict = df_dados_tratados_masc[variaveis]
y_predict = df_dados_tratados_masc[vTarget]

#%% Comparativos entre modelos

comparativo_algoritmos = pd.DataFrame(columns=['tecnica_analitica', 
                                               'tipo_base', 
                                               'acuracia', 
                                               'sensibilidade', 
                                               'especificidade', 
                                               'precisao',
                                               'VP',
                                               'FP',
                                               'VN',
                                               'FN'])

#%%######################### Árvore de Decisão ################################
###############################################################################
#%% Árvore de decisão com alguns hiperparâmetros

with open('modelos/arvore_decisao.pkl', 'rb') as f:
    tree_clf_aj = pickle.load(f)
    
# Predict na base de teste
tree_pred_test_class_aj = tree_clf_aj.predict(X_predict)
tree_pred_test_prob_aj = tree_clf_aj.predict_proba(X_predict)

# Matriz de confusão e Curva ROC
gerar_matriz_confusao('Árvore de decisão', 'Previsão', tree_pred_test_class_aj, y_predict)
gerar_curva_roc_comparacao('Árvore de decisão', 'Masculino', 'Previsão', tree_pred_train_prob_aj, y_train, tree_pred_test_prob_aj, y_predict)

#%%######################### Random Forest (Grid Search) ######################
###############################################################################
#%%

with open('modelos/random_forest.pkl', 'rb') as f:
    rf_best = pickle.load(f)
    
# Predict na base de testes
rf_grid_pred_test_class = rf_best.predict(X_predict)
rf_grid_pred_test_prob = rf_best.predict_proba(X_predict)

# Matriz de confusão (base de teste)
gerar_matriz_confusao('Random Forest (Grid Search)', 'Previsão', rf_grid_pred_test_class, y_predict)
gerar_curva_roc_comparacao('Random Forest (Grid Search)', 'Masculino', 'Previsão', rf_grid_pred_train_prob, y_train, rf_grid_pred_test_prob, y_predict)

#%%######################### XGBoost com Grid Search ##########################
###############################################################################
#%%

with open('modelos/xgboost.pkl', 'rb') as f:
    xgb_best = pickle.load(f)

# Valores preditos na base de teste
xgb_grid_pred_test_class = xgb_best.predict(X_predict)
xgb_grid_pred_test_prob = xgb_best.predict_proba(X_predict)

# Matriz de confusão e Curva ROC
gerar_matriz_confusao('XGBoost (Grid Search)', 'Previsão', xgb_grid_pred_test_class, y_predict)
gerar_curva_roc_comparacao('XGBoost (Grid Search)', 'Masculino', 'Previsão', xgb_grid_pred_train_prob, y_train, xgb_grid_pred_test_prob, y_predict)

#%%######################### SVM com Grid Search ##############################
###############################################################################
#%%

with open('modelos/svm.pkl', 'rb') as f:
    svm_best = pickle.load(f)

svm_grid_test = svm_best.predict(X_predict)
svm_grid_test_proba = svm_best.predict_proba(X_predict)

# Matriz de confusão e Curva ROC
gerar_matriz_confusao('SVM (Grid Search)', 'Previsão', svm_grid_test, y_predict)
gerar_curva_roc_comparacao('SVM (Grid Search)', 'Masculino', 'Previsão', svm_grid_train_proba, y_train, svm_grid_test_proba, y_predict)

#%%######################### K-Nearest Neighbors ##############################
###############################################################################
#%%

with open('modelos/knn.pkl', 'rb') as f:
    knn_pipe = pickle.load(f)
    
knn_pred_test = knn_pipe.predict(X_predict)
knn_pred_test_proba = knn_pipe.predict_proba(X_predict)

# Matriz de confusão e Curva ROC
gerar_matriz_confusao('K-Nearest Neighbors', 'Previsão', knn_pred_test, y_predict)
gerar_curva_roc_comparacao('K-Nearest Neighbors', 'Masculino', 'Previsão', knn_pred_train_proba, y_train, knn_pred_test_proba, y_predict)

#%%######################### Redes Neurais MLP ################################
###############################################################################
#%%

with open('modelos/rede_neural_mlp.pkl', 'rb') as f:
    mlp_pipe = pickle.load(f)
    
mlp_pred_grid_test = mlp_pipe.predict(X_predict)
mlp_pred_grid_test_proba = mlp_pipe.predict_proba(X_predict)

# Matriz de confusão e Curva ROC
gerar_matriz_confusao('Rede Neural MLP', 'Previsão', mlp_pred_grid_test, y_predict)
gerar_curva_roc_comparacao('Rede Neural MLP', 'Masculino', 'Previsão', mlp_pred_grid_train_proba, y_train, mlp_pred_grid_test_proba, y_predict)

#%%
print(comparativo_algoritmos)
comparativo_algoritmos.to_csv('resultados/estatisticas_algoritmos_masculino.csv', index=False, encoding="utf-8")
