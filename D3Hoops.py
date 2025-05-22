# Imports and grabbing data
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from io import StringIO
import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# import random

source = requests.get('https://masseyratings.com/scores.php?s=604302&sub=11620&all=1&mode=2&format=1')

webpage = soup(source.content, features='html.parser')

string = webpage.prettify()
games_csv_string = StringIO(string)
games_24_raw = pd.read_csv(games_csv_string, header=None, names=['days', 'date', 'team_1_id', 'team_1_hfa', 'team_1_score', 'team_2_id',
                                                                 'team_2_hfa',
                                                                 'team_2_score'])
games_24_raw = games_24_raw[games_24_raw['date'] != 20241102].reset_index(drop=True)

teams_24 = pd.read_csv('teams_24.txt', header=None, names=['team_id', 'team'])
conferences = pd.read_csv('d3_conference.csv')

games_1 = pd.merge(games_24_raw, teams_24, left_on='team_1_id', right_on='team_id', how='left')
games_1 = pd.merge(games_1, conferences, on='team')
games_1 = games_1.drop(['team_1_id', 'team_id'], axis=1)
games_1 = games_1.rename(columns={'team': 'team_1'})
games_2 = pd.merge(games_1, teams_24, left_on='team_2_id', right_on='team_id', how='left')
games_2 = pd.merge(games_2, conferences, on='team')
games_2 = games_2.drop(['team_2_id', 'team_id'], axis=1)
games_2 = games_2.rename(columns={'team': 'team_2'})
games_2['season'] = 2025
games_24 = games_2[['season', 'date', 'team_1', 'team_1_hfa', 'team_1_score', 'team_2', 'team_2_hfa', 'team_2_score']]

games_24['date'] = pd.to_datetime(games_24['date'].astype(str), format='%Y%m%d')
min_date = pd.to_datetime(games_24['date'].min())

for i in range(len(games_24)):
    games_24.loc[i, 'week'] = int(pd.Timedelta(games_24.loc[i, 'date'] - min_date).days / 7) + 1

games_prev = pd.read_csv('ncaa_games.csv')
games_prev = games_prev[games_prev['season'] != 2021].reset_index(drop=True)
games_prev['date'] = pd.to_datetime(games_prev['date'])
games = pd.concat([games_prev, games_24])
games = games.sort_values(by='date', ascending=True)
games = games.reset_index(drop=True)

games.to_csv('ncaa_games.csv', index=False)

# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] ==
#                                              'Washington StL'), ['team_1_hfa', 'team_2_hfa']] = [0, 0]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'NYU'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'Trinity CT'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'Emory'), ['team_1_hfa', 'team_2_hfa']] = [0, 0]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'Catholic'), ['team_1_hfa', 'team_2_hfa']] = [0, 0]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'Redlands'), ['team_1_hfa', 'team_2_hfa']] = [0, 0]
# games.loc[(games['date'] == '2025-03-14') & (games['team_1'] == 'Wesleyan CT'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-15') & (games['team_1'] == 'Wesleyan CT'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-15') & (games['team_1'] == 'Trinity CT'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-15') & (games['team_1'] == 'NYU'), ['team_1_hfa', 'team_2_hfa']] = [1, -1]
# games.loc[(games['date'] == '2025-03-15') & (games['team_1'] ==
#                                              'Washington StL'), ['team_1_hfa', 'team_2_hfa']] = [-1, 1]

# Use when wanting to test and not include the current season
# games = pd.read_csv('ncaa_games.csv')
# games = games[games['season'] != 2021].reset_index(drop=True)
# games['date'] = pd.to_datetime(games['date'])
# %%
# Insert all teams into dictionary
team_names = set()
for team in games['team_2'].unique():
    team_names.add(team)

# Adding teams when necessary
# team_names.add('PSU-Brandywine')
# %%
# Creating the team class


class NCAATeam:
    def __init__(self, name):
        self.team_name = name
        self.off_rating = 73.5
        self.def_rating = 73.5
        self.rating = 0
        self.wins = 0
        self.losses = 0
        self.off_last_year = 73.5
        self.off_2_year = 73.5
        self.off_3_year = 73.5
        self.def_last_year = 73.5
        self.def_2_year = 73.5
        self.def_3_year = 73.5


priors = pd.read_excel('Massey Priors.xlsx')
priors_2022 = pd.read_excel('Massey Priors 2022.xlsx')
# %%
# Creating the elo calculator


class EloCalculator:

    def __init__(self, hfa=1.287, init_k=0.152, k_min=0.048, k_week=0.0088,
                 mov_max=42.8, mov_min=1.9, init_k_2022=.169, k_late=0.048, k_cons=.81):
        self.mov_max = mov_max
        self.mov_min = mov_min
        self.hfa = hfa
        self.init_k = init_k
        self.k_min = k_min
        self.k_week = k_week
        self.init_k_2022 = init_k_2022
        self.k_late = k_late
        self.k_cons = k_cons

    def predict_score_1(self, scores, teams):
        if scores['team_1_hfa'] == 0:
            off_1 = teams[scores['team_1']].off_rating
            def_2 = teams[scores['team_2']].def_rating
        elif scores['team_1_hfa'] == 1:
            off_1 = teams[scores['team_1']].off_rating + (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating + (self.hfa / 2)
        else:
            off_1 = teams[scores['team_1']].off_rating - (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating - (self.hfa / 2)
        return 73.5 + (off_1 - 73.5) + (def_2 - 73.5)

    def predict_score_2(self, scores, teams):
        if scores['team_2_hfa'] == 0:
            off_2 = teams[scores['team_2']].off_rating
            def_1 = teams[scores['team_1']].def_rating
        elif scores['team_2_hfa'] == 1:
            off_2 = teams[scores['team_2']].off_rating + (self.hfa / 2)
            def_1 = teams[scores['team_1']].def_rating + (self.hfa / 2)
        else:
            off_2 = teams[scores['team_2']].off_rating - (self.hfa / 2)
            def_1 = teams[scores['team_1']].def_rating - (self.hfa / 2)
        return 73.5 + (off_2 - 73.5) + (def_1 - 73.5)

    def win_probability(self, scores, teams):
        if scores['team_1_hfa'] == 0:
            rating_1 = teams[scores['team_1']].rating
            rating_2 = teams[scores['team_2']].rating
        elif scores['team_1_hfa'] == 1:
            rating_1 = teams[scores['team_1']].rating + self.hfa
            rating_2 = teams[scores['team_2']].rating - self.hfa
        else:
            rating_1 = teams[scores['team_1']].rating - self.hfa
            rating_2 = teams[scores['team_2']].rating + self.hfa
        return 1 / (1 + np.exp(-(.15 * (rating_1 - rating_2))))

    def postgame_wp(self, scores, teams):
        return 1 / (1 + np.exp(-(.15 * (scores['team_1_score'] - scores['team_2_score']))))

    def get_k_mult(self, scores, teams):
        k = self.init_k - (self.k_week * scores['week'])
        x = self.init_k_2022 - (self.k_week * scores['week'])
        if scores['week'] >= 17:
            return self.k_late
        elif ((scores['season'] == 2022) & (x < self.k_min)):
            return self.k_min
        elif scores['season'] == 2022:
            return x
        elif k < self.k_min:
            return self.k_min
        else:
            return k

    def get_k(self, scores, teams):
        mult = self.get_k_mult(scores, teams)
        wp = self.win_probability(scores, teams)
        post_wp = self.postgame_wp(scores, teams)
        return mult * (self.k_cons + 2 * (1-self.k_cons) * min((wp + post_wp), ((1-wp) + (1-post_wp))))

    def update_single_game(self, scores, teams):
        if scores['team_1_hfa'] == 0:
            def_1 = teams[scores['team_1']].def_rating
            off_2 = teams[scores['team_2']].off_rating
            def_2 = teams[scores['team_2']].def_rating
            off_1 = teams[scores['team_1']].off_rating
        elif scores['team_1_hfa'] == 1:
            def_1 = teams[scores['team_1']].def_rating - (self.hfa / 2)
            off_2 = teams[scores['team_2']].off_rating - (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating + (self.hfa / 2)
            off_1 = teams[scores['team_1']].off_rating + (self.hfa / 2)
        else:
            def_1 = teams[scores['team_1']].def_rating + (self.hfa / 2)
            off_2 = teams[scores['team_2']].off_rating + (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating - (self.hfa / 2)
            off_1 = teams[scores['team_1']].off_rating - (self.hfa / 2)

        if scores['season'] == 2025:
            teams[scores['team_1']].wins += 1
            teams[scores['team_2']].losses += 1

        spread = (73.5 + (off_1 - 73.5) + (def_2 - 73.5)) - (73.5 + (off_2 - 73.5) + (def_1 - 73.5))

        if (scores['team_1_score'] - scores['team_2_score']) > max(self.mov_max, spread):
            team_1_score = scores['team_1_score'] - (((scores['team_1_score'] - scores['team_2_score']) - max(self.mov_max, spread))
                                                     / 2)
            team_2_score = scores['team_2_score'] + (((scores['team_1_score'] - scores['team_2_score']) - max(self.mov_max, spread))
                                                     / 2)
        elif (scores['team_1_score'] - scores['team_2_score']) < self.mov_min:
            team_1_score = scores['team_1_score'] + ((self.mov_min - (scores['team_1_score'] - scores['team_2_score']))
                                                     / 2)
            team_2_score = scores['team_2_score'] - ((self.mov_min - (scores['team_1_score'] - scores['team_2_score']))
                                                     / 2)
        else:
            team_1_score = scores['team_1_score']
            team_2_score = scores['team_2_score']

        teams[scores['team_1']].off_rating += (self.get_k(scores, teams) *
                                               (team_1_score - (73.5 + (off_1 - 73.5) + (def_2 - 73.5))))
        teams[scores['team_2']].off_rating += (self.get_k(scores, teams) *
                                               (team_2_score - (73.5 + (off_2 - 73.5) + (def_1 - 73.5))))
        teams[scores['team_1']].def_rating += (self.get_k(scores, teams) *
                                               (team_2_score - (73.5 + (off_2 - 73.5) + (def_1 - 73.5))))
        teams[scores['team_2']].def_rating += (self.get_k(scores, teams) *
                                               (team_1_score - (73.5 + (off_1 - 73.5) + (def_2 - 73.5))))

        teams[scores['team_1']].rating = teams[scores['team_1']].off_rating - teams[scores['team_1']].def_rating
        teams[scores['team_2']].rating = teams[scores['team_2']].off_rating - teams[scores['team_2']].def_rating


# %%
# Setting up rating calculation
teams = {}
for team in team_names:
    teams[team] = NCAATeam(team)
    teams[team].off_rating = priors.loc[priors['Team'] == team, 'JacobOff'].values[0]
    teams[team].def_rating = priors.loc[priors['Team'] == team, 'JacobDef'].values[0]
    teams[team].rating = priors.loc[priors['Team'] == team, 'JacobRating'].values[0]
elo = EloCalculator()
n_games = games.shape[0]
seasons = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025]
# %%
# Rating calculation
for year in seasons:
    for team in teams:
        if (year == 2022) and (pd.isna(priors_2022.loc[priors_2022['Team'] == team, 'JacobOff'].values[0]) is False):
            teams[team].off_3_year = teams[team].off_last_year
            teams[team].off_2_year = teams[team].off_rating
            teams[team].off_last_year = priors_2022.loc[priors_2022['Team'] == team, 'JacobOff'].values[0]
            teams[team].def_3_year = teams[team].def_last_year
            teams[team].def_2_year = teams[team].def_rating
            teams[team].def_last_year = priors_2022.loc[priors_2022['Team'] == team, 'JacobDef'].values[0]
        else:
            teams[team].off_3_year = teams[team].off_2_year
            teams[team].off_2_year = teams[team].off_last_year
            teams[team].off_last_year = teams[team].off_rating
            teams[team].def_3_year = teams[team].def_2_year
            teams[team].def_2_year = teams[team].def_last_year
            teams[team].def_last_year = teams[team].def_rating
        if year >= 2012:
            teams[team].off_rating = (.102 * teams[team].off_3_year) + \
                (.796 * teams[team].off_last_year) + (.102 * teams[team].off_2_year)
            teams[team].def_rating = (.102 * teams[team].def_3_year) + \
                (.796 * teams[team].def_last_year) + (.102 * teams[team].def_2_year)
            teams[team].rating = teams[team].off_rating - teams[team].def_rating
    for i in range(n_games):
        if games.loc[i, 'season'] == year:
            # games.loc[i, 'team_1_rating'] = teams[games.loc[i, 'team_1']].rating
            # games.loc[i, 'team_2_rating'] = teams[games.loc[i, 'team_2']].rating
            # games.loc[i, 'error'] = np.abs((games.loc[i, 'team_1_score'] - games.loc[i, 'team_2_score']) -
            #                                (elo.predict_score_1(games.iloc[i], teams) - elo.predict_score_2(games.iloc[i], teams)))
            # games.loc[i, 'spread'] = elo.predict_score_1(
            #     games.iloc[i], teams) - elo.predict_score_2(games.iloc[i], teams)
            # games.loc[i, 'k'] = elo.get_k(games.iloc[i], teams)
            elo.update_single_game(games.iloc[i], teams)
# %%
# Error checking
this_yr = games[games['season'] == 2025]
this_yr['error'].mean()
group = this_yr[['date', 'error']].groupby('date').mean()
week_error = this_yr[['week', 'error']].groupby('week').mean()
week_error_full = games[['week', 'error']].groupby('week').mean()
week_error_full['count'] = games[['week', 'error']].groupby('week').count()

last_yr = games[games['season'] == 2024]
last_yr['error'].mean()
last_yr_week = last_yr[['week', 'error']].groupby('week').mean()
last_yr_week['count'] = last_yr[['week', 'error']].groupby('week').count()

season_error = games[['season', 'error']].groupby('season').mean()
games['error'].mean()

games['mov'] = games['team_1_score'] - games['team_2_score']
mov_group = games[['season', 'mov']].groupby('season').mean()
# %%
# spread_check = this_yr[['team_1_score', 'team_2_score', 'spread']]
# spread_check['mov'] = np.where(spread_check['spread'] < 0, spread_check['team_2_score'] -
#                                spread_check['team_1_score'], spread_check['team_1_score'] - spread_check['team_2_score'])
# spread_check['spread_rd'] = np.abs(round(spread_check['spread'] * 2) / 2)
# spread_check['error'] = spread_check['spread'] - spread_check['mov']
# spread_group = spread_check.groupby('spread_rd').mean().reset_index()

# sns.scatterplot(data=spread_group, x='spread_rd', y='mov')
# plt.axline((0, 0), slope=1, color='k', ls='--')
# %%
# wp_check = this_yr[['spread']].reset_index(drop=True)
# for i in range(len(wp_check)):
#     if wp_check.loc[i, 'spread'] > 0:
#         wp_check.loc[i, 'win'] = 1
#     else:
#         wp_check.loc[i, 'win'] = 0

# wp_check['spread_rd'] = np.abs(round(wp_check['spread'] / 2) * 2)
# wp_check_group = wp_check[['spread_rd', 'win']].groupby('spread_rd').mean()

# x = np.linspace(0, 40)
# y = 1 / (1 + np.exp(-.15 * (x)))

# sns.scatterplot(data=wp_check_group, x='spread_rd', y='win', color='purple')
# plt.plot(x, y, color='k', ls='--')
# plt.title('2024-25 D3 MBB Estimated & Observed WP')
# plt.xlabel('Predicted Spread (Rounded To Nearest 2 Points)')
# plt.ylabel('Est. WP (Line) & Obs. WP (Dots)')
# %%
# rating_graph = pd.DataFrame()
# team_graph = games[(games['team_1'] == 'Wheaton IL') | (games['team_2'] == 'Wheaton IL')].reset_index(drop=True)
# for i in range(len(team_graph)):
#     rating_graph.loc[i, 'game'] = i
#     rating_graph.loc[i, 'season'] = team_graph.loc[i, 'season']
#     rating_graph.loc[i, 'team'] = 'Wheaton IL'
#     if team_graph.loc[i, 'team_1'] == 'Wheaton IL':
#         rating_graph.loc[i, 'team_rating'] = team_graph.loc[i, 'team_1_rating']
#         rating_graph.loc[i, 'opponent'] = team_graph.loc[i, 'team_2']
#     else:
#         rating_graph.loc[i, 'team_rating'] = team_graph.loc[i, 'team_2_rating']
#         rating_graph.loc[i, 'opponent'] = team_graph.loc[i, 'team_1']

# sns.lineplot(data=rating_graph, x='game', y='team_rating')
# plt.ylim(-25, 25)  # Set x-axis limits to range from -25 to 25
# plt.axhline(y=15, color='r', linestyle='--', label='Top 25 Caliber')  # Dotted line for 15
# plt.axhline(y=0, color='b', linestyle='--', label='Average')  # Dotted line for 0

# # Add labels and legend
# plt.xlabel('Game')
# plt.ylabel('Team Rating')
# plt.title('Team Rating Over Time')
# plt.legend()
# %%
# Ratings
ratings = pd.DataFrame(columns=['team', 'record', 'off', 'def', 'rating'])
for team in teams.keys():
    ratings.loc[team] = pd.Series(
        {'team': team,
         'record': str(teams[team].wins) + '-' + str(teams[team].losses),
         'off': teams[team].off_rating,
         'def': teams[team].def_rating,
         'rating': teams[team].rating})
teams_24 = pd.read_csv('teams_24.txt', header=None, names=['team_id', 'team'])
ratings = pd.merge(teams_24, ratings, on='team', how='left')
ratings = ratings.drop('team_id', axis=1)
ratings = pd.merge(ratings, conferences, on='team')
ratings = ratings.sort_values(by='rating', ascending=False)
ratings = ratings.reset_index(drop=True)
ratings['rank'] = ratings.index + 1
ratings = ratings.set_index('rank', drop=True)
ratings = ratings[['ncaa_name', 'conference', 'record', 'rating', 'off', 'def']]
ratings[['rating', 'off', 'def']] = ratings[['rating', 'off', 'def']].astype(float)

ratings.to_csv('d3_ratings.csv')
# %%
cons = ratings[['conference', 'rating']].groupby('conference').mean()
# %%
# Testing
# testing = pd.DataFrame(columns=['hfa', 'init_k', 'k_min', 'k_week', 'mov_max', 'mov_min', 'init_k_2022', 'k_late',
#                                 'prior_weight', 'k_cons'])
# testing['hfa'] = np.random.uniform(1.206, 1.298, 100)
# testing['init_k'] = np.random.uniform(.151, .158, 100)
# testing['k_min'] = np.random.uniform(.046, .054, 100)
# testing['k_week'] = np.random.uniform(.0085, .0094, 100)
# testing['mov_max'] = np.random.uniform(35, 45, 100)
# testing['mov_min'] = np.random.uniform(0.4, 3.2, 100)
# testing['init_k_2022'] = np.random.uniform(.163, .182, 100)
# testing['k_late'] = np.random.uniform(.046, .06, 100)
# testing['prior_weight'] = np.random.uniform(.758, .799, 100)
# testing['k_cons'] = np.random.uniform(.77, .91, 100)
# testing[['hfa', 'init_k', 'k_min', 'init_k_2022', 'k_late', 'prior_weight']] = round(
#     testing[['hfa', 'init_k', 'k_min', 'init_k_2022', 'k_late', 'prior_weight']], 3)
# testing['k_week'] = round(testing['k_week'], 4)
# testing[['mov_max', 'mov_min']] = round(testing[['mov_max', 'mov_min']], 1)
# testing['k_cons'] = round(testing['k_cons'], 2)

# for test in range(len(testing)):
#     teams = {}
#     for team in team_names:
#         teams[team] = NCAATeam(team)
#         teams[team].off_rating = priors.loc[priors['Team'] == team, 'JacobOff'].values[0]
#         teams[team].def_rating = priors.loc[priors['Team'] == team, 'JacobDef'].values[0]
#         teams[team].rating = priors.loc[priors['Team'] == team, 'JacobRating'].values[0]
#     elo = EloCalculator(hfa=testing.loc[test, 'hfa'], init_k=testing.loc[test, 'init_k'],
#                         k_min=testing.loc[test, 'k_min'], k_week=testing.loc[test, 'k_week'],
#                         mov_max=testing.loc[test, 'mov_max'], mov_min=testing.loc[test, 'mov_min'],
#                         init_k_2022=testing.loc[test, 'init_k_2022'], k_late=testing.loc[test, 'k_late'],
#                         k_cons=testing.loc[test, 'k_cons'])
#     error = 0
#     for year in seasons:
#         for team in teams:
#             if (year == 2022) and (pd.isna(priors_2022.loc[priors_2022['Team'] == team, 'JacobOff'].values[0]) is False):
#                 teams[team].off_3_year = teams[team].off_last_year
#                 teams[team].off_2_year = teams[team].off_rating
#                 teams[team].off_last_year = priors_2022.loc[priors_2022['Team'] == team, 'JacobOff'].values[0]
#                 teams[team].def_3_year = teams[team].def_last_year
#                 teams[team].def_2_year = teams[team].def_rating
#                 teams[team].def_last_year = priors_2022.loc[priors_2022['Team'] == team, 'JacobDef'].values[0]
#             else:
#                 teams[team].off_3_year = teams[team].off_2_year
#                 teams[team].off_2_year = teams[team].off_last_year
#                 teams[team].off_last_year = teams[team].off_rating
#                 teams[team].def_3_year = teams[team].def_2_year
#                 teams[team].def_2_year = teams[team].def_last_year
#                 teams[team].def_last_year = teams[team].def_rating
#             if year >= 2012:
#                 teams[team].off_rating = (((1-testing.loc[test, 'prior_weight']) / 2) * teams[team].off_3_year) + \
#                     (testing.loc[test, 'prior_weight'] * teams[team].off_last_year) + \
#                     (((1-testing.loc[test, 'prior_weight']) / 2) * teams[team].off_2_year)
#                 teams[team].def_rating = (((1-testing.loc[test, 'prior_weight']) / 2) * teams[team].def_3_year) + \
#                     (testing.loc[test, 'prior_weight'] * teams[team].def_last_year) + \
#                     (((1-testing.loc[test, 'prior_weight']) / 2) * teams[team].def_2_year)
#         for i in range(n_games):
#             if games.loc[i, 'season'] == year:
#                 error += np.abs((elo.predict_score_1(games.iloc[i], teams) - elo.predict_score_2(
#                     games.iloc[i], teams)) - (games.loc[i, 'team_1_score'] - games.loc[i, 'team_2_score']))
#                 elo.update_single_game(games.iloc[i], teams)
#     avg_error = error / n_games
#     testing.loc[test, 'error'] = avg_error
#     print(test, ": ", avg_error)
# best = testing[testing['error'] < 9.4]
# 9.395937844485509

# %%
# Game Score Check
# game_score = games[['team_1_rating', 'team_2_rating', 'spread']]
# game_score['ratings'] = game_score['team_1_rating'] + game_score['team_2_rating']
# game_score['abs_spread'] = np.abs(game_score['spread'])
# game_score['ratings'].mean()
# game_score['ratings'].std()
# game_score['abs_spread'].mean()
# game_score['abs_spread'].std()
# %%
# Grab and clean future games
source_fut = requests.get('https://masseyratings.com/scores.php?s=604302&sub=11620&all=1&mode=2&sch=on&format=1')

webpage_fut = soup(source_fut.content, features='html.parser')

string_fut = webpage_fut.prettify()
StringIO_fut = StringIO(string_fut)
fut_games = pd.read_csv(StringIO_fut, header=None, names=['days', 'date', 'team_1_id', 'team_1_hfa', 'team_1_score', 'team_2_id',
                                                          'team_2_hfa', 'team_2_score'])
fut_games = fut_games[fut_games['team_1_score'] == 0]

games_1_f = pd.merge(fut_games, teams_24, left_on='team_1_id', right_on='team_id', how='left')
games_1_f = games_1_f.drop(['team_1_id', 'team_id'], axis=1)
games_1_f = games_1_f.rename(columns={'team': 'team_1'})
games_2_f = pd.merge(games_1_f, teams_24, left_on='team_2_id', right_on='team_id', how='left')
games_2_f = games_2_f.drop(['team_2_id', 'team_id'], axis=1)
games_2_f = games_2_f.rename(columns={'team': 'team_2'})
games_24_f = games_2_f[['date', 'team_1', 'team_1_hfa', 'team_1_score', 'team_2', 'team_2_hfa', 'team_2_score']]
games_24_f['date'] = pd.to_datetime(games_24_f['date'].astype(str), format='%Y%m%d')

games_24_f[['team_1_hfa', 'team_2_hfa']] = [0, 0]
# %%
# Score and predict future games
for i in range(len(games_24_f)):
    if games_24_f.loc[i, 'team_1_hfa'] == -1:
        games_24_f.loc[i, 'hfa'] = '@'
    elif games_24_f.loc[i, 'team_1_hfa'] == 1:
        games_24_f.loc[i, 'hfa'] = 'vs'
    else:
        games_24_f.loc[i, 'hfa'] = 'neu'
    games_24_f.loc[i, 'team_1_pred_score'] = elo.predict_score_1(games_24_f.iloc[i], teams)
    games_24_f.loc[i, 'team_2_pred_score'] = elo.predict_score_2(games_24_f.iloc[i], teams)

games_24_f['spread'] = games_24_f['team_1_pred_score'] - games_24_f['team_2_pred_score']

for i in range(len(games_24_f)):
    games_24_f.loc[i, 'ratings'] = teams[games_24_f.loc[i, 'team_1']].rating + teams[games_24_f.loc[i, 'team_2']].rating

games_24_f['ratings_z'] = (games_24_f['ratings'] - 0.9318203758738816) / 14.893169166413983
games_24_f['spread_z'] = (8.592811967140756 - np.abs(games_24_f['spread'])) / 6.564897231299327
games_24_f['game_score'] = (games_24_f['ratings_z'] + games_24_f['spread_z']) / 2

games_24_f = pd.merge(games_24_f, conferences, left_on='team_1', right_on='team')
games_24_f = games_24_f.drop(['team_1'], axis=1)
games_24_f = games_24_f.rename(columns={'ncaa_name': 'team_1'})
games_24_f = pd.merge(games_24_f, conferences, left_on='team_2', right_on='team')
games_24_f = games_24_f.drop(['team_2'], axis=1)
games_24_f = games_24_f.rename(columns={'ncaa_name': 'team_2'})

games_24_f = games_24_f[games_24_f['date'] >= pd.Timestamp(datetime.date.today())]
games_24_f = games_24_f.sort_values(by=['date', 'game_score'], ascending=[True, False]).reset_index(drop=True)

games_24_f = games_24_f[['date', 'team_1', 'team_1_pred_score', 'hfa', 'team_2',
                         'team_2_pred_score', 'spread']]

games_24_f.to_csv('future_games.csv', index=False)
