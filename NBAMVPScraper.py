import pandas as pd
# Library for opening url and creating requests
import urllib.request
# for parsing all the tables present on the website
from html_table_parser.parser import HTMLTableParser
# %%


def url_get_contents(url):
    # making request to the website
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    # reading contents of the website
    return f.read()


# %%


def get_bball_df(website):
    # defining the html contents of a URL.
    xhtml = url_get_contents(website).decode('utf-8')

# Defining the HTMLTableParser object
    p = HTMLTableParser()

# feeding the html contents in the HTMLTableParser object
    p.feed(xhtml)

    df = pd.DataFrame(p.tables[0])

    df.columns = df.loc[0]
    df = df.query("Rk != 'Rk'")
    return df


# %%
advanced = get_bball_df(website='https://www.basketball-reference.com/leagues/NBA_2025_advanced.html')
ppg = get_bball_df('https://www.basketball-reference.com/leagues/NBA_2025_per_game.html')

advanced = advanced.drop_duplicates(subset='Player')
ppg = ppg.drop_duplicates(subset='Player')
# %%
teams_xhtml = url_get_contents('https://www.basketball-reference.com/leagues/NBA_2025.html').decode('utf-8')

# Defining the HTMLTableParser object
teams_p = HTMLTableParser()

# feeding the html contents in the HTMLTableParser object
teams_p.feed(teams_xhtml)

teams = pd.DataFrame(teams_p.tables[10])
# %%
teams.columns = teams.loc[1]
teams.dropna(inplace=True)
teams = teams.loc[2:31]
teams['Team'] = teams['Team'].str.replace(' *', '')
# %%
abbs = pd.read_csv('Team Summaries.csv')
abbs.query('season == 2023', inplace=True)
abbs = abbs[['team', 'abbreviation']]

team_sums = pd.merge(teams, abbs, left_on='Team', right_on='team', how='inner')
# %%
advanced = advanced[['Player', 'Team', 'G', 'PER', 'WS', 'MP']]
ppg = ppg[['Player', 'Team', 'PTS', 'TRB', 'AST']]
# ppg = ppg.rename(columns={'Team': 'Tm'})
team_sums = team_sums[['abbreviation', 'W', 'L']]
# %%
player_stats = pd.merge(advanced, ppg, on=['Player', 'Team'])
# %%
nba = pd.merge(player_stats, team_sums, left_on='Team', right_on='abbreviation', how='outer')
nba = nba.drop(['abbreviation'], axis=1)
nba['share'] = 0
nba['season'] = 2025
nba['winner'] = 'FALSE'
# %%
nba = nba.replace('', 0)
nba = nba.astype({'G': int,
                  'PER': float,
                  'WS': float,
                  'MP': int,
                  'PTS': float,
                  'TRB': float,
                  'AST': float})
nba = nba.query('MP > 499')
# %%
nba = nba.set_index('Player')
# %%
nba = nba.drop(['MP'], axis=1)
nba.columns = ['tm', 'g', 'per', 'ws', 'pts_per_game', 'trb_per_game',
               'ast_per_game', 'w', 'l', 'share', 'season', 'winner']
# %%
nba.to_csv('NBA_24.csv', encoding='utf-8-sig')
