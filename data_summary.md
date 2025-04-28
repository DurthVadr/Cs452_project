# NBA Data Exploration Summary

## Dataset Overview

### game_info
- Shape: (11979, 13)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, away_elo_i, away_elo_n, home_elo_i, home_elo_n, elo_pred
- Missing values: 0

### team_stats
- Shape: (23958, 37)
- Columns: game_id, team, MP, FG, FGA, FGp, 3P, 3PA, 3Pp, FT, FTA, FTp, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, PM, TSp, eFGp, 3PAr, FTr, ORBp, DRBp, TRBp, ASTp, STLp, BLKp, TOVp, USGp, ORtg, DRtg, BPM
- Missing values: 47916

### team_factor_10
- Shape: (11979, 16)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_eFGp, a_FTr, a_ORBp, a_TOVp, h_eFGp, h_FTr, h_ORBp, h_TOVp
- Missing values: 124

### team_factor_20
- Shape: (11979, 16)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_eFGp, a_FTr, a_ORBp, a_TOVp, h_eFGp, h_FTr, h_ORBp, h_TOVp
- Missing values: 124

### team_factor_30
- Shape: (11979, 16)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_eFGp, a_FTr, a_ORBp, a_TOVp, h_eFGp, h_FTr, h_ORBp, h_TOVp
- Missing values: 124

### team_full_10
- Shape: (11979, 66)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_FG, a_FGA, a_FGp, a_3P, a_3PA, a_3Pp, a_FT, a_FTA, a_FTp, a_ORB, a_DRB, a_TRB, a_AST, a_STL, a_BLK, a_TOV, a_PF, a_PTS, a_TSp, a_eFGp, a_3PAr, a_FTr, a_ORBp, a_DRBp, a_TRBp, a_ASTp, a_STLp, a_BLKp, a_TOVp, h_FG, h_FGA, h_FGp, h_3P, h_3PA, h_3Pp, h_FT, h_FTA, h_FTp, h_ORB, h_DRB, h_TRB, h_AST, h_STL, h_BLK, h_TOV, h_PF, h_PTS, h_TSp, h_eFGp, h_3PAr, h_FTr, h_ORBp, h_DRBp, h_TRBp, h_ASTp, h_STLp, h_BLKp, h_TOVp
- Missing values: 899

### team_full_20
- Shape: (11979, 66)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_FG, a_FGA, a_FGp, a_3P, a_3PA, a_3Pp, a_FT, a_FTA, a_FTp, a_ORB, a_DRB, a_TRB, a_AST, a_STL, a_BLK, a_TOV, a_PF, a_PTS, a_TSp, a_eFGp, a_3PAr, a_FTr, a_ORBp, a_DRBp, a_TRBp, a_ASTp, a_STLp, a_BLKp, a_TOVp, h_FG, h_FGA, h_FGp, h_3P, h_3PA, h_3Pp, h_FT, h_FTA, h_FTp, h_ORB, h_DRB, h_TRB, h_AST, h_STL, h_BLK, h_TOV, h_PF, h_PTS, h_TSp, h_eFGp, h_3PAr, h_FTr, h_ORBp, h_DRBp, h_TRBp, h_ASTp, h_STLp, h_BLKp, h_TOVp
- Missing values: 899

### team_full_30
- Shape: (11979, 66)
- Columns: game_id, season, date, away_team, away_score, home_team, home_score, result, a_FG, a_FGA, a_FGp, a_3P, a_3PA, a_3Pp, a_FT, a_FTA, a_FTp, a_ORB, a_DRB, a_TRB, a_AST, a_STL, a_BLK, a_TOV, a_PF, a_PTS, a_TSp, a_eFGp, a_3PAr, a_FTr, a_ORBp, a_DRBp, a_TRBp, a_ASTp, a_STLp, a_BLKp, a_TOVp, h_FG, h_FGA, h_FGp, h_3P, h_3PA, h_3Pp, h_FT, h_FTA, h_FTp, h_ORB, h_DRB, h_TRB, h_AST, h_STL, h_BLK, h_TOV, h_PF, h_PTS, h_TSp, h_eFGp, h_3PAr, h_FTr, h_ORBp, h_DRBp, h_TRBp, h_ASTp, h_STLp, h_BLKp, h_TOVp
- Missing values: 899

### nbaallelo
- Shape: (126314, 24)
- Columns: Unnamed: 0, gameorder, game_id, lg_id, _iscopy, year_id, date_game, seasongame, is_playoffs, team_id, fran_id, pts, elo_i, elo_n, win_equiv, opp_id, opp_fran, opp_pts, opp_elo_i, opp_elo_n, game_location, game_result, forecast, notes
- Missing values: 120890

## 2018-2019 Season Analysis

- Number of games: 1230
- Date range: 2018-10-16 to 2019-04-10

## Home Court Advantage

- Home team win percentage: 57.3%
- Away team win percentage: 42.7%

## ELO Rating Analysis

- ELO prediction accuracy: 65.3%
- Average home team initial ELO: 1505.6
- Average away team initial ELO: 1505.0

## Upset Analysis

- Upset rate: 35.6%
- Total upsets: 4268 out of 11979 games

## Four Factors Analysis

- Missing values in 10-game average: 124
- Missing values in 20-game average: 124
- Missing values in 30-game average: 124

## Team Statistics Analysis

- Number of unique teams: 30
- Top 5 scoring teams: GSW, HOU, LAC, DEN, POR

