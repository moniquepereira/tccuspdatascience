# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 16:29:21 2026

@author: nique
"""

#!pip install selenium
#!pip install webdriver_manager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

"""Funções para leitura da URL"""

WAIT_SECONDS = 12.0  # adjust if needed

def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    #driver = webdriver.Chrome(options=opts)
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=opts)
    return driver

"""
Buscar resultados das partidas por set
"""
def buscar_resultados_partida(soup, game_id, tournament):
  dados_partida = []

  season = soup.find("div", {"class": "vbw-mu__info--season"}).text
  phase = soup.find("div", {"class": "vbw-mu__info--details"}).text

  div_home = soup.find("div",  {"class" : "vbw-mu__team vbw-mu__team--home"})
  nome_time_home = div_home.find("div", {"class":"vbw-mu__team__name"}).text

  resultado_geral = soup.find("div", {"class": "vbw-mu__sets"})
  dados_sets = resultado_geral.find_all("div", {"class" : "vbw-mu__sets--result"})

  lista_dados = []
  for s in dados_sets:
      if int(s['data-set-no']) > 5:
        continue

      regex = re.compile('vbw-mu__pointA*')
      points_teamA = s.find("span", {"class" :regex}).text

      if len(points_teamA) == 0:
        continue

      lista_dados.append([game_id, nome_time_home, s['data-set-no'], points_teamA])

  div_away = soup.find("div",  {"class" : "vbw-mu__team vbw-mu__team--away"})
  nome_time_away = div_away.find("div", {"class":"vbw-mu__team__name"}).text

  for s in dados_sets:
      if int(s['data-set-no']) > 5:
        continue
      regex = re.compile('vbw-mu__pointB')
      points_teamB = s.find("span", {"class" :regex}).text

      if len(points_teamB) == 0:
        continue

      lista_dados.append([game_id, nome_time_away, s['data-set-no'], points_teamB])

  dados_partida.append([game_id, season, tournament, phase, nome_time_home, nome_time_away])

  return dados_partida, lista_dados, nome_time_home, nome_time_away

"""
Buscar estatísticas de uma ação do jogo
  attack - ataques efetuados por jogador
  block - bloqueios efetuados por jogador
  serve - saques efetuados por jogador
  reception - recepções efetuadas por jogador
  dig - defesas efetuadas por jogador
  set - levantamentos efetuados por jogador
"""
def buscar_estatisticas_acao(soup, desc_acao, id_partida):
  list = []
  times = ['teama', 'teamb']

  for desc_time in times:
      regex = re.compile('vbw-o-table vbw-match-player-statistic-table vbw-stats-*'+desc_acao+'*')

      tabelas_dados = soup.find_all("table", {"class" : regex,
                                              "data-team": desc_time,
                                              "data-stattype": desc_acao})

      for tabela in tabelas_dados:
          if tabela['data-set'] == 'all':
            continue

          table_rows = tabela.find_all('tr')

          for tr in table_rows:
              row = [id_partida, tabela['data-team'], tabela['data-set'], tabela['data-stattype']]

              for td in tr.find_all('td'):
                  row.append(td.text)

              list.append(row)

  return pd.DataFrame(list)

def main():
    dados_competicoes = [{"COMPETICAO":"VNL","ANO":2021,"CATEGORIA":"M","PARTIDA_INI":11700,"PARTIDA_FIM":11823,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2021/schedule/"},
                         {"COMPETICAO":"VNL","ANO":2022,"CATEGORIA":"M","PARTIDA_INI":13650,"PARTIDA_FIM":13753,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2022/schedule/"},
                         {"COMPETICAO":"VNL","ANO":2023,"CATEGORIA":"M","PARTIDA_INI":16128,"PARTIDA_FIM":16231,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2023/schedule/"},
                         {"COMPETICAO":"VNL","ANO":2024,"CATEGORIA":"M","PARTIDA_INI":18853,"PARTIDA_FIM":18956,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2024/schedule/"},
                         {"COMPETICAO":"VNL","ANO":2025,"CATEGORIA":"M","PARTIDA_INI":21437,"PARTIDA_FIM":21552,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/2025/schedule/"},
                         {"COMPETICAO":"WC","ANO":2022,"CATEGORIA":"M","PARTIDA_INI":13455,"PARTIDA_FIM":13506,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/men-world-championship/2022/schedule/"},
                         {"COMPETICAO":"WC","ANO":2025,"CATEGORIA":"M","PARTIDA_INI":21062,"PARTIDA_FIM":21125,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/men-world-championship/schedule/"},
                         {"COMPETICAO":"OG","ANO":2021,"CATEGORIA":"M","PARTIDA_INI":11344,"PARTIDA_FIM":11381,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/olympics-2020/schedule/"},
                         {"COMPETICAO":"OG","ANO":2024,"CATEGORIA":"M","PARTIDA_INI":19061,"PARTIDA_FIM":19086,"URL_BASE":"https://en.volleyballworld.com/volleyball/competitions/volleyball-olympic-games-paris-2024/schedule/"}]

    dados_partida_todas = pd.DataFrame()
    dados_partida_sets  = pd.DataFrame()

    dados_saque = pd.DataFrame()
    dados_defesa = pd.DataFrame()
    dados_bloqueio = pd.DataFrame()
    dados_recepcao = pd.DataFrame()
    dados_levantamento = pd.DataFrame()
    dados_ataque = pd.DataFrame()

    for competicao in dados_competicoes:
        for id_partida in range(competicao["PARTIDA_INI"],competicao["PARTIDA_FIM"]+1):
            driver = make_driver()

            url_partida = competicao['URL_BASE']+str(id_partida)

            try:
                print("Loading page...")
                print(url_partida)
                driver.get(url_partida)
                time.sleep(WAIT_SECONDS)  # espera enquanto JS está carregando
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
            finally:
                driver.quit()

            try:
                dados_partida, dados_sets, teamA, teamB = buscar_resultados_partida(soup, id_partida, competicao['COMPETICAO'])
                print(dados_partida)
                print(dados_sets)

                dados_partida_todas = pd.concat([dados_partida_todas, pd.DataFrame(dados_partida)])
                dados_partida_sets  = pd.concat([dados_partida_sets, pd.DataFrame(dados_sets)])

                dados_saque_jogo = buscar_estatisticas_acao(soup, 'serve', id_partida)
                dados_saque_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_saque = pd.concat([dados_saque, dados_saque_jogo], ignore_index=True)

                dados_defesa_jogo = buscar_estatisticas_acao(soup, 'dig', id_partida)
                dados_defesa_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_defesa = pd.concat([dados_defesa, dados_defesa_jogo], ignore_index=True)

                dados_bloqueio_jogo = buscar_estatisticas_acao(soup, 'block', id_partida)
                dados_bloqueio_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_bloqueio = pd.concat([dados_bloqueio, dados_bloqueio_jogo], ignore_index=True)

                dados_recepcao_jogo = buscar_estatisticas_acao(soup, 'reception', id_partida)
                dados_recepcao_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_recepcao = pd.concat([dados_recepcao, dados_recepcao_jogo], ignore_index=True)

                dados_levantamento_jogo = buscar_estatisticas_acao(soup, 'set', id_partida)
                dados_levantamento_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_levantamento = pd.concat([dados_levantamento, dados_levantamento_jogo], ignore_index=True)

                dados_ataque_jogo = buscar_estatisticas_acao(soup, 'attack', id_partida)
                dados_ataque_jogo.replace({'teama':teamA, 'teamb':teamB}, inplace=True)
                dados_ataque = pd.concat([dados_ataque, dados_ataque_jogo], ignore_index=True)
            finally:
                continue

    dados_saque.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'point', 'error', 'attempts', 'total', 'efficiency']
    dados_saque.dropna(subset=['player_name'], inplace=True)
    dados_saque.to_csv('saque_masc.csv')
    dados_saque.info()

    dados_defesa.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'digs', 'errors', 'total', 'efficiency']
    dados_defesa.dropna(subset=['player_name'], inplace=True)
    dados_defesa.to_csv('defesa_masc.csv')
    dados_defesa.info()

    dados_bloqueio.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'point', 'errors', 'touches', 'total', 'efficiency']
    dados_bloqueio.dropna(subset=['player_name'], inplace=True)
    dados_bloqueio.to_csv('bloqueio_masc.csv')
    dados_bloqueio.info()

    dados_recepcao.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'successful', 'errors', 'attempts', 'total', 'efficiency']
    dados_recepcao.dropna(subset=['player_name'], inplace=True)
    dados_recepcao.to_csv('recepcao_masc.csv')
    dados_recepcao.info()

    dados_levantamento.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'point', 'errors', 'attempts', 'total', 'efficiency']
    dados_levantamento.dropna(subset=['player_name'], inplace=True)
    dados_levantamento.to_csv('levantamento_masc.csv')
    dados_levantamento.info()

    dados_ataque.columns = ['game_id','team', 'set_nr', 'action', 'player_nr', 'player_name', 'position', 'point', 'errors', 'attempts', 'total', 'efficiency']
    dados_ataque.dropna(subset=['player_name'], inplace=True)
    dados_ataque.to_csv('ataque_masc.csv')
    dados_ataque.info()

    dados_partida_todas.to_csv('lista_jogos_masc.csv')

    dados_partida_sets.to_csv('lista_jogos_sets_masc.csv')

if __name__ == "__main__":
    main()

