# Build a simple interactive dashboard for the uploaded IPL matches dataset
# - Loads CSV
# - Creates a few core charts (matches over time, winners, venues, toss decision impact)
# - Displays as an interactive Plotly dashboard in-notebook

import pandas as pd
import numpy as np
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import display, HTML

csv_path = './ipl_matches_data.csv'
ipl_df = pd.read_csv(csv_path, encoding='ascii')

# Basic cleaning / typing
ipl_df['match_date'] = pd.to_datetime(ipl_df['match_date'], dayfirst=True, errors='coerce')

num_cols = ['win_by_runs', 'win_by_wickets', 'overs', 'balls_per_over', 'match_number']
for col in num_cols:
    if col in ipl_df.columns:
        ipl_df[col] = pd.to_numeric(ipl_df[col], errors='coerce')

# Normalized helper columns
ipl_df['season'] = pd.to_numeric(ipl_df['season'], errors='coerce')
ipl_df['year'] = ipl_df['match_date'].dt.year
ipl_df['month'] = ipl_df['match_date'].dt.to_period('M').dt.to_timestamp()
ipl_df['is_result_win'] = (ipl_df['result'].astype(str).str.lower() == 'win')
ipl_df['toss_winner_won_match'] = (ipl_df['toss_winner'] == ipl_df['match_winner']) & ipl_df['match_winner'].notna()

# Display head immediately
display(ipl_df.head())

# Precompute some aggregates used by the dashboard
matches_by_month = (
    ipl_df.dropna(subset=['month'])
    .groupby('month', as_index=False)
    .agg(matches=('match_id', 'count'))
    .sort_values('month')
)

matches_by_season = (
    ipl_df.dropna(subset=['season'])
    .groupby('season', as_index=False)
    .agg(matches=('match_id', 'count'))
    .sort_values('season')
)

winner_counts = (
    ipl_df.dropna(subset=['match_winner'])
    .groupby('match_winner', as_index=False)
    .agg(wins=('match_id', 'count'))
    .sort_values('wins', ascending=False)
)

venue_counts = (
    ipl_df.dropna(subset=['venue'])
    .groupby('venue', as_index=False)
    .agg(matches=('match_id', 'count'))
    .sort_values('matches', ascending=False)
)

city_counts = (
    ipl_df.dropna(subset=['city'])
    .groupby('city', as_index=False)
    .agg(matches=('match_id', 'count'))
    .sort_values('matches', ascending=False)
)

pom_counts = (
    ipl_df.dropna(subset=['player_of_match'])
    .groupby('player_of_match', as_index=False)
    .agg(awards=('match_id', 'count'))
    .sort_values('awards', ascending=False)
)

# Toss decision impact (overall)
toss_impact = (
    ipl_df.dropna(subset=['toss_decision', 'match_winner', 'toss_winner'])
    .groupby('toss_decision', as_index=False)
    .agg(
        matches=('match_id', 'count'),
        toss_winner_match_wins=('toss_winner_won_match', 'sum'),
    )
)
toss_impact['toss_win_rate'] = np.where(toss_impact['matches'] > 0, toss_impact['toss_winner_match_wins'] / toss_impact['matches'], np.nan)

# KPI cards
kpi_total_matches = int(ipl_df['match_id'].nunique())
kpi_total_seasons = int(ipl_df['season'].dropna().nunique()) if 'season' in ipl_df.columns else int(ipl_df['season_id'].nunique())
kpi_total_venues = int(ipl_df['venue'].dropna().nunique())
kpi_total_cities = int(ipl_df['city'].dropna().nunique())

# Create charts
fig_matches_season = px.bar(
    matches_by_season,
    x='season', y='matches',
    title='Matches per Season',
)
fig_matches_season.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))

fig_matches_month = px.line(
    matches_by_month,
    x='month', y='matches',
    markers=True,
    title='Matches Over Time (Monthly)',
)
fig_matches_month.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))

top_n = 12
fig_top_winners = px.bar(
    winner_counts.head(top_n),
    x='wins', y='match_winner',
    orientation='h',
    title='Top Teams by Match Wins (Top ' + str(top_n) + ')'
)
fig_top_winners.update_layout(height=420, yaxis={'categoryorder': 'total ascending'}, margin=dict(l=20, r=20, t=50, b=20))

fig_top_venues = px.bar(
    venue_counts.head(top_n),
    x='matches', y='venue',
    orientation='h',
    title='Top Venues by Matches Hosted (Top ' + str(top_n) + ')'
)
fig_top_venues.update_layout(height=420, yaxis={'categoryorder': 'total ascending'}, margin=dict(l=20, r=20, t=50, b=20))

fig_toss = px.bar(
    toss_impact,
    x='toss_decision', y='toss_win_rate',
    text=toss_impact['toss_win_rate'].map(lambda v: ('' if pd.isna(v) else str(round(100*v, 1)) + '%')),
    title='Toss Decision vs Toss-Winner Match Win Rate',
)
fig_toss.update_yaxes(tickformat='.0%')
fig_toss.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))

# Build an interactive filter panel using a little HTML + JS and Plotly JSON
# For speed and simplicity: we create a single Plotly figure with dropdown filters (season, city, team)

filter_df = ipl_df.copy()
filter_df['team1'] = filter_df['team1'].astype(str)
filter_df['team2'] = filter_df['team2'].astype(str)
filter_df['match_winner'] = filter_df['match_winner'].astype(str)

# A compact figure showing win margin distributions and toss-win share for the filtered slice
# Precompute for each season to support dropdown without needing a full JS data pipeline

def slice_metrics(df_slice):
    total_m = int(df_slice['match_id'].nunique())
    win_rate_toss = df_slice['toss_winner_won_match'].mean() if len(df_slice) else np.nan
    avg_runs = df_slice['win_by_runs'].dropna().mean() if 'win_by_runs' in df_slice.columns else np.nan
    avg_wkts = df_slice['win_by_wickets'].dropna().mean() if 'win_by_wickets' in df_slice.columns else np.nan
    return total_m, win_rate_toss, avg_runs, avg_wkts

seasons_sorted = [int(x) for x in sorted(ipl_df['season'].dropna().unique())]

season_to_payload = {}
for s in seasons_sorted:
    df_s = filter_df[filter_df['season'] == s]
    total_m, win_rate_toss, avg_runs, avg_wkts = slice_metrics(df_s)
    # winner counts for the season
    wc = (
        df_s[df_s['match_winner'].notna() & (df_s['match_winner'] != 'nan')]
        .groupby('match_winner', as_index=False)
        .agg(wins=('match_id', 'count'))
        .sort_values('wins', ascending=False)
        .head(10)
    )
    season_to_payload[str(s)] = {
        'kpis': {
            'matches': total_m,
            'toss_win_rate': win_rate_toss,
            'avg_win_by_runs': avg_runs,
            'avg_win_by_wkts': avg_wkts,
        },
        'winner_bar': wc
    }

# Default season = most recent
default_season = str(seasons_sorted[-1]) if len(seasons_sorted) else None

def build_winner_fig(winner_bar_df, season_label):
    if winner_bar_df is None or len(winner_bar_df) == 0:
        fig = go.Figure()
        fig.update_layout(title='Top Winners (Season ' + str(season_label) + ')', height=320)
        return fig
    fig = px.bar(
        winner_bar_df,
        x='wins', y='match_winner',
        orientation='h',
        title='Top Winners (Season ' + str(season_label) + ')'
    )
    fig.update_layout(height=320, yaxis={'categoryorder': 'total ascending'}, margin=dict(l=20, r=20, t=50, b=20))
    return fig

initial_winner_fig = build_winner_fig(season_to_payload[default_season]['winner_bar'], default_season) if default_season else go.Figure()

# Build HTML dashboard
kpi_html = """
<div style='display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;'>
  <div style='flex:1; min-width:180px; padding:12px; border:1px solid #e5e7eb; border-radius:10px;'>
    <div style='font-size:12px; color:#6b7280;'>Total matches</div>
    <div style='font-size:24px; font-weight:700;'>{tm}</div>
  </div>
  <div style='flex:1; min-width:180px; padding:12px; border:1px solid #e5e7eb; border-radius:10px;'>
    <div style='font-size:12px; color:#6b7280;'>Seasons</div>
    <div style='font-size:24px; font-weight:700;'>{ts}</div>
  </div>
  <div style='flex:1; min-width:180px; padding:12px; border:1px solid #e5e7eb; border-radius:10px;'>
    <div style='font-size:12px; color:#6b7280;'>Venues</div>
    <div style='font-size:24px; font-weight:700;'>{tv}</div>
  </div>
  <div style='flex:1; min-width:180px; padding:12px; border:1px solid #e5e7eb; border-radius:10px;'>
    <div style='font-size:12px; color:#6b7280;'>Cities</div>
    <div style='font-size:24px; font-weight:700;'>{tc}</div>
  </div>
</div>
""".format(tm=kpi_total_matches, ts=kpi_total_seasons, tv=kpi_total_venues, tc=kpi_total_cities)

season_options_html = "".join(["<option value='" + str(s) + "'>" + str(s) + "</option>" for s in seasons_sorted])

# Serialize payload
import json
season_payload_json = json.dumps({
    'default_season': default_season,
    'season_to_payload': {
        k: {
            'kpis': v['kpis'],
            'winner_bar': v['winner_bar'].to_dict(orient='list')
        } for k, v in season_to_payload.items()
    }
})

# Convert figures to HTML snippets
fig1_html = fig_matches_season.to_html(include_plotlyjs=True, full_html=False)
fig2_html = fig_matches_month.to_html(include_plotlyjs=False, full_html=False)
fig3_html = fig_top_winners.to_html(include_plotlyjs=False, full_html=False)
fig4_html = fig_top_venues.to_html(include_plotlyjs=False, full_html=False)
fig5_html = fig_toss.to_html(include_plotlyjs=False, full_html=False)
fig6_html = initial_winner_fig.to_html(include_plotlyjs=False, full_html=False)

html_out = """
<div style='font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 6px;'>
  <div style='margin-bottom:8px;'>
    <div style='font-size:20px; font-weight:800;'>IPL Matches Dashboard (Basics)</div>
    <div style='color:#6b7280; font-size:12px;'>Quick look at match volume, winners, venues, and toss impact.</div>
  </div>

  """ + kpi_html + """

  <div style='display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; margin: 12px 0 8px 0;'>
    <div style='min-width:260px;'>
      <div style='font-size:12px; color:#6b7280; margin-bottom:4px;'>Season drilldown</div>
      <select id='season_select' style='padding:8px; border:1px solid #d1d5db; border-radius:8px; min-width:220px;'>
        """ + season_options_html + """
      </select>
    </div>

    <div style='display:flex; gap:10px; flex-wrap:wrap;'>
      <div style='padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; min-width:210px;'>
        <div style='font-size:12px; color:#6b7280;'>Matches in season</div>
        <div id='kpi_matches' style='font-size:20px; font-weight:800;'>-</div>
      </div>
      <div style='padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; min-width:210px;'>
        <div style='font-size:12px; color:#6b7280;'>Toss-winner match win rate</div>
        <div id='kpi_toss' style='font-size:20px; font-weight:800;'>-</div>
      </div>
      <div style='padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; min-width:210px;'>
        <div style='font-size:12px; color:#6b7280;'>Avg win by runs</div>
        <div id='kpi_runs' style='font-size:20px; font-weight:800;'>-</div>
      </div>
      <div style='padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; min-width:210px;'>
        <div style='font-size:12px; color:#6b7280;'>Avg win by wickets</div>
        <div id='kpi_wkts' style='font-size:20px; font-weight:800;'>-</div>
      </div>
    </div>
  </div>

  <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;'>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>""" + fig1_html + """</div>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>""" + fig2_html + """</div>
  </div>

  <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;'>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>""" + fig3_html + """</div>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>""" + fig4_html + """</div>
  </div>

  <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>""" + fig5_html + """</div>
    <div style='border:1px solid #eef2f7; border-radius:10px; padding:6px;'>
      <div id='season_winner_chart'>""" + fig6_html + """</div>
    </div>
  </div>
</div>

<script>
  const payload = """ + season_payload_json + """;

  function fmtPct(v){
    if(v === null || v === undefined || isNaN(v)) return '-';
    return (100*v).toFixed(1) + '%';
  }
  function fmtNum(v){
    if(v === null || v === undefined || isNaN(v)) return '-';
    return Number(v).toFixed(1);
  }

  function updateKPIs(seasonKey){
    const d = payload.season_to_payload[seasonKey];
    if(!d) return;
    document.getElementById('kpi_matches').textContent = d.kpis.matches;
    document.getElementById('kpi_toss').textContent = fmtPct(d.kpis.toss_win_rate);
    document.getElementById('kpi_runs').textContent = fmtNum(d.kpis.avg_win_by_runs);
    document.getElementById('kpi_wkts').textContent = fmtNum(d.kpis.avg_win_by_wkts);
  }

  function updateWinnerChart(seasonKey){
    const d = payload.season_to_payload[seasonKey];
    const container = document.getElementById('season_winner_chart');
    if(!d){
      container.innerHTML = '';
      return;
    }
    const wc = d.winner_bar;
    const yVals = wc.match_winner || [];
    const xVals = wc.wins || [];

    const trace = {
      type: 'bar',
      orientation: 'h',
      y: yVals,
      x: xVals,
      marker: {color: '#636EFA'}
    };
    const layout = {
      title: {text: 'Top Winners (Season ' + seasonKey + ')'},
      height: 320,
      margin: {l: 20, r: 20, t: 50, b: 20},
      yaxis: {categoryorder: 'total ascending'}
    };

    container.innerHTML = "<div id='winner_plot'></div>";
    Plotly.newPlot('winner_plot', [trace], layout, {displayModeBar: false, responsive: true});
  }

  const seasonSelect = document.getElementById('season_select');
  if(payload.default_season){
    seasonSelect.value = payload.default_season;
    updateKPIs(payload.default_season);
    updateWinnerChart(payload.default_season);
  }

  seasonSelect.addEventListener('change', (e) => {
    const s = e.target.value;
    updateKPIs(s);
    updateWinnerChart(s);
  });
</script>
"""

display(HTML(html_out))