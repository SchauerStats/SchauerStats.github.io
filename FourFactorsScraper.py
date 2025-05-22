import seaborn as sns
import pandas as pd
import urllib.request
from html_table_parser.parser import HTMLTableParser
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
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

    df = pd.DataFrame(p.tables[2])

    df.columns = df.loc[1]
    return df
# %%


def get_poss_data(year_id, team_id, team_name):
    advanced = get_bball_df(
        website='https://stats.ncaa.org/player/game_by_game?game_sport_year_ctl_id=' + year_id + '&org_id=' + team_id + '&stats_player_seq=-100')
    advanced = advanced.drop(0)
    advanced = advanced[advanced['G'] != '']
    advanced = advanced[advanced['Opponent'] != 'Opponent']
    advanced['OppDRebs'] = advanced['DRebs'].shift(-1)
    advanced = advanced[advanced['Opponent'] != 'Defensive Totals'].reset_index(drop=True)
    advanced = advanced[advanced['Date'].str.startswith('*Contest') == False]
    advanced = advanced[['Opponent', 'FGM', 'FGA', '3FG', 'FT', 'FTA', 'ORebs', 'OppDRebs', 'TO']]
    advanced = advanced.replace('', 0)
    advanced[['FGM', 'FGA', '3FG', 'FT', 'FTA', 'ORebs', 'OppDRebs', 'TO']] = advanced[[
        'FGM', 'FGA', '3FG', 'FT', 'FTA', 'ORebs', 'OppDRebs', 'TO']].replace('/', '', regex=True)
    advanced[['FGM', 'FGA', '3FG', 'FT', 'FTA', 'ORebs', 'OppDRebs', 'TO']] = advanced[[
        'FGM', 'FGA', '3FG', 'FT', 'FTA', 'ORebs', 'OppDRebs', 'TO']].astype(int)
    advanced['EFG'] = (advanced['FGM'] + (0.5 * advanced['3FG'])) / advanced['FGA']
    advanced['TORate'] = advanced['TO'] / (advanced['FGA'] + (.44 * advanced['FTA']) + advanced['TO'])
    advanced['ORBRate'] = advanced['ORebs'] / (advanced['ORebs'] + advanced['OppDRebs'])
    advanced['FTR'] = advanced['FT'] / advanced['FGA']
    advanced['Team'] = team_name
    advanced['TeamID'] = team_id
    for i in range(len(advanced)):
        if advanced.loc[i, 'Opponent'].startswith('@'):
            advanced.loc[i, 'HFA'] = -1
            advanced.loc[i, 'Opponent'] = advanced.loc[i, 'Opponent'].replace('@ ', '')
        elif '@' in advanced.loc[i, 'Opponent']:
            advanced.loc[i, 'HFA'] = 0
            advanced.loc[i, 'Opponent'] = advanced.loc[i, 'Opponent'].split(' @')[0]
        else:
            advanced.loc[i, 'HFA'] = 1
    return advanced


teams = pd.read_excel('Poss2022TeamID.xlsx')
teams[['NCAA ID', 'Year Code']] = teams[['NCAA ID', 'Year Code']].astype(str)
games = pd.DataFrame()
for i in range(len(teams)):
    team_data = get_poss_data(teams.loc[i, 'Year Code'], teams.loc[i, 'NCAA ID'], teams.loc[i, 'Team'])
    games = pd.concat([games, team_data])
    print(str(i) + ': ' + teams.loc[i, 'Team'])
games.to_excel('FourFactors2223.xlsx', index=False)
# %%
games = pd.read_excel('FourFactors2223.xlsx')
games = games[['Team', 'Opponent', 'EFG', 'TORate', 'ORBRate', 'FTR']]
teams = pd.read_excel('TeamID2425.xlsx')
games = pd.merge(games, teams[['NCAATeam', 'DatacastTeam']], left_on='Opponent', right_on='NCAATeam', how='inner')
# %%


def get_team_ratings(games, column):
    x = games[['Team', 'Opponent']]
    x = pd.get_dummies(x)
    y = column

    reg = Ridge(fit_intercept=True)
    reg.fit(X=x, y=y)

    coef = reg.coef_ + reg.intercept_
    coef = pd.DataFrame(coef.T)
    cols = pd.DataFrame(x.columns.values)

    results = pd.concat([cols, coef], axis=1)
    results.columns = ['team', 'rating']

    results = results[results['team'].str.startswith("Team")]
    results['team'] = results['team'].str.replace('Team_', '')

    ratings = results.sort_values(by='rating', ascending=False)
    ratings = ratings.reset_index(drop=True)
    ratings['rank'] = ratings.index
    ratings = ratings.set_index('rank', drop=True)
    return ratings


def get_opp_ratings(games, column):
    x = games[['Team', 'Opponent']]
    x = pd.get_dummies(x)
    y = column

    reg = Ridge(fit_intercept=True)
    reg.fit(X=x, y=y)

    coef = reg.coef_ + reg.intercept_
    coef = pd.DataFrame(coef.T)
    cols = pd.DataFrame(x.columns.values)

    results = pd.concat([cols, coef], axis=1)
    results.columns = ['team', 'rating']

    results = results[results['team'].str.startswith("Opponent")]
    results['team'] = results['team'].str.replace('Opponent_', '')

    ratings = results.sort_values(by='rating', ascending=False)
    ratings = ratings.reset_index(drop=True)
    ratings['rank'] = ratings.index
    ratings = ratings.set_index('rank', drop=True)
    return ratings
# %%


ratings = pd.DataFrame()

efg_ratings = get_team_ratings(games, games['EFG'])
to_ratings = get_team_ratings(games, games['TORate'])
orb_ratings = get_team_ratings(games, games['ORBRate'])
ftr_ratings = get_team_ratings(games, games['FTR'])
efg_opp_ratings = get_opp_ratings(games, games['EFG'])
to_opp_ratings = get_opp_ratings(games, games['TORate'])
orb_opp_ratings = get_opp_ratings(games, games['ORBRate'])
ftr_opp_ratings = get_opp_ratings(games, games['FTR'])

ratings = pd.merge(efg_ratings, to_ratings, on='team')
ratings.columns = ['team', 'EFG', 'TOR']
ratings = pd.merge(ratings, orb_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'ORB'})
ratings = pd.merge(ratings, ftr_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'FTR'})
ratings = pd.merge(ratings, efg_opp_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'OEFG'})
ratings = pd.merge(ratings, to_opp_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'OTOR'})
ratings = pd.merge(ratings, orb_opp_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'OORB'})
ratings = pd.merge(ratings, ftr_opp_ratings, on='team')
ratings = ratings.rename(columns={'rating': 'OFTR'})
ratings['EFGRk'] = ratings['EFG'].rank(ascending=False)
ratings['TORRk'] = ratings['TOR'].rank()
ratings['ORBRk'] = ratings['ORB'].rank(ascending=False)
ratings['FTRRk'] = ratings['FTR'].rank(ascending=False)
ratings['OEFGRk'] = ratings['OEFG'].rank()
ratings['OTORRk'] = ratings['OTOR'].rank(ascending=False)
ratings['OORBRk'] = ratings['OORB'].rank()
ratings['OFTRRk'] = ratings['OFTR'].rank()

ratings = pd.merge(ratings, teams, left_on='team', right_on='Team')
ratings = ratings[['DatacastTeam', 'Conference', 'EFG', 'TOR', 'ORB', 'FTR', 'OEFG', 'OTOR', 'OORB', 'OFTR', 'EFGRk',
                   'TORRk', 'ORBRk', 'FTRRk', 'OEFGRk', 'OTORRk', 'OORBRk', 'OFTRRk']]
ratings = ratings.rename(columns={'DatacastTeam': 'Team'})
# %%
eff = pd.read_excel('Efficiency2425.xlsx')
ratings = pd.merge(ratings, eff, on=['Team', 'Conference'])
# %%
off_x = ratings[['EFG', 'TOR', 'ORB', 'FTR']]
off_y = ratings['AdjO']
def_x = ratings[['OEFG', 'OTOR', 'OORB', 'OFTR']]
def_x.columns = ['EFG', 'TOR', 'ORB', 'FTR']
def_y = ratings['AdjD']
def_y.columns = ['AdjO']
x = pd.concat([off_x, def_x])
y = pd.concat([off_y, def_y])

mod = LinearRegression()
mod.fit(x, y)
ratings['PredO'] = mod.predict(off_x)
ratings['PredD'] = mod.predict(def_x)
r2_score(ratings['AdjO'], ratings['PredO'])
r2_score(ratings['AdjD'], ratings['PredD'])
# %%
coef = mod.coef_
feature_names = ['EFG', 'TOR', 'ORB', 'FTR']
coef_df = pd.DataFrame([coef], columns=feature_names)
# %%
ratings['EFGPts'] = ratings['EFG'] * coef_df.loc[0, 'EFG']
ratings['TORPts'] = ratings['TOR'] * coef_df.loc[0, 'TOR']
ratings['ORBPts'] = ratings['ORB'] * coef_df.loc[0, 'ORB']
ratings['FTRPts'] = ratings['FTR'] * coef_df.loc[0, 'FTR']
ratings['OEFGPts'] = ratings['OEFG'] * coef_df.loc[0, 'EFG']
ratings['OTORPts'] = ratings['OTOR'] * coef_df.loc[0, 'TOR']
ratings['OORBPts'] = ratings['OORB'] * coef_df.loc[0, 'ORB']
ratings['OFTRPts'] = ratings['OFTR'] * coef_df.loc[0, 'FTR']

ratings['EFGPtsAA'] = ratings['EFGPts'] - ratings['EFGPts'].mean()
ratings['TORPtsAA'] = ratings['TORPts'] - ratings['TORPts'].mean()
ratings['ORBPtsAA'] = ratings['ORBPts'] - ratings['ORBPts'].mean()
ratings['FTRPtsAA'] = ratings['FTRPts'] - ratings['FTRPts'].mean()
ratings['OEFGPtsAA'] = ratings['OEFGPts'].mean() - ratings['OEFGPts']
ratings['OTORPtsAA'] = ratings['OTORPts'].mean() - ratings['OTORPts']
ratings['OORBPtsAA'] = ratings['OORBPts'].mean() - ratings['OORBPts']
ratings['OFTRPtsAA'] = ratings['OFTRPts'].mean() - ratings['OFTRPts']
ratings['ShootPtsAA'] = ratings['EFGPtsAA'] + ratings['OEFGPtsAA']
ratings['TOPtsAA'] = ratings['TORPtsAA'] + ratings['OTORPtsAA']
ratings['RBPtsAA'] = ratings['ORBPtsAA'] + ratings['OORBPtsAA']
ratings['FTPtsAA'] = ratings['FTRPtsAA'] + ratings['OFTRPtsAA']
ratings['PredEM'] = ratings['EFGPtsAA'] + ratings['TORPtsAA'] + ratings['ORBPtsAA'] + ratings['FTRPtsAA'] + \
    ratings['OEFGPtsAA'] + ratings['OTORPtsAA'] + ratings['OORBPtsAA'] + ratings['OFTRPtsAA']

ratings.to_excel('FourFactorRatings2425.xlsx', index=False)
