# -*- coding: utf-8 -*-
"""
Created on Sat Jan 03 08:25:32 2026

@author: nique
"""

#%% Importar pacotes de tratamento

import pandas as pd
import numpy as np
from pathlib import Path

#%% Preparação dos dados para aplicação nos algoritmos

# Arquivos de entrada e saída
INPUT_SETS   = "lista_jogos_sets.csv"
INPUT_JOGOS  = "lista_jogos.csv"
OUTPUT       = "df_consolidado_saida.csv"

# Leitura do arquivo de pontuações para obter vencedores por set
df_pontuacao_sets = pd.read_csv(INPUT_SETS, skiprows=1, header=None, encoding='utf-8')
df_pontuacao_sets = df_pontuacao_sets.rename(columns={1: "ID_PARTIDA", 2: "TIME", 3: "ID_SET", 4: "PONTOS"})
df_pontuacao_sets["ID_PARTIDA"] = df_pontuacao_sets["ID_PARTIDA"].astype(str)
df_pontuacao_sets["ID_SET"]     = df_pontuacao_sets["ID_SET"].astype(str)
df_pontuacao_sets["PONTOS"]     = pd.to_numeric(df_pontuacao_sets["PONTOS"], errors="coerce").fillna(0).astype(int)
idx = df_pontuacao_sets.groupby(["ID_PARTIDA", "ID_SET"])["PONTOS"].idxmax()
vencedores = df_pontuacao_sets.loc[idx, ["ID_PARTIDA", "ID_SET", "TIME"]].copy().rename(columns={"TIME": "VENCEDOR"})
vencedores = vencedores.sort_values(["ID_PARTIDA", "ID_SET"]).reset_index(drop=True)

# Leitura do arquivo de jogos para obter se jogo ocorreu em fase final de competição
df_dados_partidas = pd.read_csv(INPUT_JOGOS, skiprows=1, header=None, encoding='utf-8')
df_dados_partidas = df_dados_partidas.rename(columns={1: "ID_PARTIDA", 2: "ANO_COMPETICAO", 3: "NOME_COMPETICAO", 4: "DESC_FASE"})
df_dados_partidas["ID_PARTIDA"] = df_dados_partidas["ID_PARTIDA"].astype(str)
df_dados_partidas = df_dados_partidas.drop(df_dados_partidas.columns[[0,5,6]], axis=1)
df_dados_partidas['FASE_FINAL'] = df_dados_partidas['DESC_FASE'].apply(lambda x: True if 'Final' in x else False)

# Arquivos com dados das ações de jogo
FILES = {
    "RECEPCAO":     Path("recepcao.csv"),
    "SAQUE":        Path("saque.csv"),
    "LEVANTAMENTO": Path("levantamento.csv"),
    "DEFESA":       Path("defesa.csv"),
    "BLOQUEIO":     Path("bloqueio.csv"),
    "ATAQUE":       Path("ataque.csv"),
}

COL_MAP = {
    "RECEPCAO":     ("total", "efficiency"),
    "SAQUE":        ("total", "efficiency"),
    "LEVANTAMENTO": ("total", "efficiency"),
    "DEFESA":       ("total", "efficiency"),
    "BLOQUEIO":     ("total", "efficiency"),
    "ATAQUE":       ("total", "efficiency"),
}

def read_table(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df

def trata_atributo_eficiencia(df: pd.DataFrame, eff_col_name: str):
    # Padroniza string e converte para float (trata vírgulas, percentuais, espaços, valores vazios)
    
    if eff_col_name not in df.columns:
        df[eff_col_name] = np.nan
        return df
    
    s = df[eff_col_name].astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan})
    
    # remove % e substituir vírgula por ponto
    s = s.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    
    # valores vazios ou não numéricos virarão NaN
    df[eff_col_name] = pd.to_numeric(s, errors="coerce")
    
    # tratamento das eficiências nulas quando há ação de jogo para a atleta (lacuna do scraping)
    mask_action = ((df["total"].fillna(0) > 0))
    mask_missing_eff = df[eff_col_name].isna()
    mask_fill = mask_action & mask_missing_eff
    
    if mask_fill.any():
        # calcular eficiência para as linhas selecionadas
        df.loc[mask_fill, eff_col_name] = df.loc[mask_fill].apply(calcula_eficiencia_na, axis=1)

    # limitar valores fora de -100..100 (cap)
    df[eff_col_name] = df[eff_col_name].where(df[eff_col_name].between(-100, 100), np.nan)
    
    return df

# função para calcular eficiência por linha
def calcula_eficiencia_na(linha):
    
    #total de ações executadas será o denominador
    t = linha.get("total")
    
    denom = None
    if pd.notna(t) and t > 0:
        denom = t
    else:
        return np.nan
    
    # buscar o nome do atributo que representa as ações bem sucedidas para cada ação de jogo
    col_pontos = procura_coluna_df(df, ("point", "successful", "digs"))
    pts  = linha.get(col_pontos, 0.0)
    errs = linha.get("errors", 0.0)
    
    # se pts/errs forem NaN, tratar como 0 para cálculo
    pts = 0.0 if pd.isna(pts) else pts
    errs = 0.0 if pd.isna(errs) else errs
    
    return (pts - errs) / denom * 100.0

def procura_coluna_df(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def calcula_eficiencia_ponderada(df: pd.DataFrame, total_col: str, eff_col: str):
    if df is None:
        return None
    # localizar colunas padrão
    game_col = procura_coluna_df(df, ("game_id", "game"))
    set_col  = procura_coluna_df(df, ("set_nr", "set"))
    team_col = procura_coluna_df(df, ("team", "team_name"))
    
    if game_col is None or set_col is None or team_col is None:
        raise ValueError(f"Colunas chave não encontradas em: {df.columns.tolist()}")
    
    # limpar efficiency
    df = trata_atributo_eficiencia(df, eff_col)
    # garantir coluna total numérica
    if total_col not in df.columns:
        df[total_col] = 0
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
    
    # linhas com total > 0 contribuem; se efficiency NaN, tratar como 0 na numeração
    df_valid = df[df[total_col] > 0].copy()
    if df_valid.empty:
        groups = df.groupby([game_col, set_col, team_col]).size().reset_index()[[game_col, set_col, team_col]]
        groups = groups.drop_duplicates().rename(columns={game_col:"ID_PARTIDA", set_col:"ID_SET", team_col:"TIME"})
        groups["EFF"] = np.nan
        return groups
   
    # calcular produto total * efficiency em coluna auxiliar
    df_valid = df_valid.assign(_prod = df_valid[total_col] * df_valid[eff_col].fillna(0))
           
    # agregações sem usar .apply sobre grupos
    agg_num = df_valid.groupby([game_col, set_col, team_col])["_prod"].sum().rename("num")
    agg_den = df_valid.groupby([game_col, set_col, team_col])[total_col].sum().rename("den")

    agg = pd.concat([agg_num, agg_den], axis=1).reset_index()
    agg["EFF"] = agg["num"] / agg["den"]

    agg = agg.rename(columns={game_col:"ID_PARTIDA", set_col:"ID_SET", team_col:"TIME"})
    return agg[["ID_PARTIDA","ID_SET","TIME","EFF"]]

# Processar arquivos (sem gravar intermediários)
resultados = {}
for action, path in FILES.items():
    df = read_table(path)
    
    if df is None:
        resultados[action] = None
        continue
    total_col, eff_col = COL_MAP[action]
    resultados[action] = calcula_eficiencia_ponderada(df, total_col, eff_col).rename(columns={"EFF": action})

# Construir chave única e juntar (sem salvar consolidado intermediário)
frames = [v[["ID_PARTIDA","ID_SET","TIME"]] for v in resultados.values() if v is not None]
if not frames:
    raise SystemExit("Nenhum dado disponível.")
keys = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

out = keys.copy()
for action, dfagg in resultados.items():
    if dfagg is None:
        out[action] = np.nan
    else:
        out = out.merge(dfagg, on=["ID_PARTIDA","ID_SET","TIME"], how="left")

cols = ["ID_PARTIDA","ID_SET","TIME"]
for c in ["ATAQUE","SAQUE","BLOQUEIO","RECEPCAO","LEVANTAMENTO","DEFESA"]:
    if c in out.columns:
        cols.append(c)
out = out[cols].sort_values(["ID_PARTIDA","ID_SET","TIME"]).reset_index(drop=True)

# garantir tipos iguais (string) nas chaves antes de unir os dataframes
out["ID_PARTIDA"] = out["ID_PARTIDA"].astype(str)
out["ID_SET"]     = out["ID_SET"].astype(str)

vencedores["ID_PARTIDA"] = vencedores["ID_PARTIDA"].astype(str)
vencedores["ID_SET"]     = vencedores["ID_SET"].astype(str)

# Unir com informação de fase final
df_final = pd.merge(out, df_dados_partidas, on=["ID_PARTIDA"], how="left")

# Unir com vencedores em memória e preparar saída final
df_final = pd.merge(df_final, vencedores, on=["ID_PARTIDA", "ID_SET"], how="left")
df_final["VENCEU_SET"] = (df_final["TIME"] == df_final["VENCEDOR"]).astype('category')

eff_cols = [c for c in ["ATAQUE","SAQUE","BLOQUEIO","RECEPCAO","LEVANTAMENTO","DEFESA"] if c in df_final.columns]
cols_saida = ["ID_PARTIDA", "ID_SET", "TIME"] + eff_cols + ["FASE_FINAL", "VENCEU_SET"]
dados_tratados = df_final[cols_saida]

#%% Elimina repetição juntando as linhas de informações das duas equipes

# função que transforma grupo de 2 em uma linha
def agrupar_dados_times_set(g):
    if len(g) != 2:
        # se diferente de 2, retornar None para filtrar depois
        return None
    a, b = g.iloc[0], g.iloc[1]
    row = {
        "ID_PARTIDA": a["ID_PARTIDA"],
        "ID_SET":     a["ID_SET"],
        "TIMEA":      a["TIME"],
        "TIMEB":      b["TIME"],
        "FASE_FINAL": a["FASE_FINAL"],
    }
    
    # adicionar eficiências time A e time B
    for col in eff_cols:
        row[f"{col}_A"] = a.get(col, pd.NA)
        row[f"{col}_B"] = b.get(col, pd.NA)
      
    # resultados
    row["VENCEU_SET_A"] = bool(a.get("VENCEU_SET"))
    row["VENCEU_SET_B"] = bool(b.get("VENCEU_SET"))
    return pd.Series(row)

# agrupar dados dos times adversários a uma única linha no dataframe final
groups = dados_tratados.groupby(["ID_PARTIDA","ID_SET"], sort=True)

rows = []
for _, g in groups:    
    s = agrupar_dados_times_set(g)    
    if s is not None:        
        rows.append(s)
        
df_agrupado = pd.DataFrame(rows).reset_index(drop=True)

# Buscar resultado do set anterior da partida - agregação de dado histórico
# Para o primeiro set de cada partida, o resultado será NaN, pois não há um set anterior jogado.
df_agrupado = df_agrupado.sort_values(by=['ID_PARTIDA', 'ID_SET'])
df_agrupado['VENCEU_SET_A_ANTERIOR'] = df_agrupado.groupby('ID_PARTIDA')['VENCEU_SET_A'].shift(1)
df_agrupado['VENCEU_SET_B_ANTERIOR'] = df_agrupado.groupby('ID_PARTIDA')['VENCEU_SET_B'].shift(1)

# ordenar as colunas para resultado final
cols_order = ["ID_PARTIDA","ID_SET","TIMEA","TIMEB"] + \
             [f"{c}_A" for c in eff_cols] + [f"{c}_B" for c in eff_cols] + \
             ["FASE_FINAL", "VENCEU_SET_A_ANTERIOR", "VENCEU_SET_B_ANTERIOR", "VENCEU_SET_A","VENCEU_SET_B"]
df_agrupado = df_agrupado.loc[:, [c for c in cols_order if c in df_agrupado.columns]]

df_agrupado.to_csv(OUTPUT, index=False, encoding="utf-8")