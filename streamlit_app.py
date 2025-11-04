import streamlit as st
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import statsmodels.api as sm
import numpy as np
import os
import calendar
import altair as alt
import plotly.graph_objects as go
from eia_storage_surprise_prediction_model import predict_storage_surprise

API_KEY = "yJ2qCjteXo197dckEhJ6SFihWiJTERMdufptE5XO"
PRICE_SERIES_ID = "RNGWHHD"
STORAGE_SERIES_ID = "NW2_EPG0_SWO_R48_BCF"
PRODUCTION_SERIES_ID = "N9010US2"
DB_FILE = "gas_data.db"
T_BASE = 18.0
LAT, LON = 29.7604, -95.3698  # Houston

st.title("US Gas Prices & Storage Analytics")

# ========================
# FETCH AND STORE DATA
# ========================
@st.cache_data
def fetch_prices():
    url = f"https://api.eia.gov/v2/natural-gas/pri/fut/data/?api_key={API_KEY}&frequency=daily&data[0]=value&facets[series][]={PRICE_SERIES_ID}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)[["period", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("period")
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("prices", conn, if_exists="replace", index=False)
    conn.close()
    return df

@st.cache_data
def fetch_storage():
    url = f"https://api.eia.gov/v2/natural-gas/stor/wkly/data/?api_key={API_KEY}&frequency=weekly&data[0]=value&facets[series][]={STORAGE_SERIES_ID}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])[["period", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("storage", conn, if_exists="replace", index=False)
    conn.close()
    return df

@st.cache_data
def fetch_production():
    url = f"https://api.eia.gov/v2/natural-gas/prod/sum/data/?api_key={API_KEY}&frequency=monthly&data[0]=value&facets[series][]={PRODUCTION_SERIES_ID}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["response"]["data"])[["period", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("production", conn, if_exists="replace", index=False)
    conn.close()
    return df

# ========================
# TEMPERATURE & FORECAST FUNCTIONS
# ========================
@st.cache_data
def fetch_temperature(start_date, end_date, lat=LAT, lon=LON, t_base=T_BASE):
    base_hist = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "America/Chicago"
    }
    r = requests.get(base_hist, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "tmax": data["daily"]["temperature_2m_max"],
        "tmin": data["daily"]["temperature_2m_min"]
    })
    df["date"] = pd.to_datetime(df["date"])
    df['tavg'] = (df['tmax'] + df['tmin']) / 2
    df['HDD'] = np.maximum(t_base - df['tavg'], 0)
    df['CDD'] = np.maximum(df['tavg'] - t_base, 0)
    return df

@st.cache_data
def fetch_temperature_forecast(start_date, end_date, lat=LAT, lon=LON, t_base=T_BASE):
    base_fc = "https://seasonal-api.open-meteo.com/v1/seasonal"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "America/Chicago"
    }
    r = requests.get(base_fc, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "tmax": data["daily"]["temperature_2m_max"],
        "tmin": data["daily"]["temperature_2m_min"]
    })
    df["date"] = pd.to_datetime(df["date"])
    df['tavg'] = (df['tmax'] + df['tmin']) / 2
    df['HDD'] = np.maximum(t_base - df['tavg'], 0)
    df['CDD'] = np.maximum(df['tavg'] - t_base, 0)
    return df

# ========================
# DISPLAY DATA
# ========================
st.sidebar.header("Data Options")
# number of prior years to include when computing historical percentiles (applies to prices & storage)
hist_years = st.sidebar.slider("Years of historical data to use for percentiles", 1, 20, 5, 1)
show_prices = st.sidebar.checkbox("Gas Prices")
show_storage = st.sidebar.checkbox("Gas Storage")
show_production = st.sidebar.checkbox("Gas Production")
show_temp_stats = st.sidebar.checkbox("Temperature Stats")

# New: separate toggles for Monte Carlo and Regression forecasts
show_mc_forecast = st.sidebar.checkbox("Monte Carlo Storage Forecast")
show_regression_forecast = st.sidebar.checkbox("Regression Storage Forecast")

# New checkbox for percentile-ratio view
show_percentile_ratio = st.sidebar.checkbox("Percentile Ratio (Storage & Price)")

# helper: percentile rank (0-100) using exclusive ties handling similar to common definition
def percentile_rank(arr, value):
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.nan
    less = np.sum(arr < value)
    equal = np.sum(arr == value)
    return (less + 0.5 * equal) / arr.size * 100.0

if st.button("Fetch Latest Data"):
    with st.spinner("Fetching prices, storage and temperature data..."):
        # fetch and persist prices & storage (cached)
        df_prices = fetch_prices()
        df_storage = fetch_storage()
        df_production = fetch_production()

        # fetch temperature history & forecast (cached) if storage returned rows
        try:
            if not df_storage.empty:
                start_hist = df_storage['period'].min() - pd.Timedelta(days=7*5*52)
                end_hist = df_storage['period'].max()
                st.session_state['df_hist'] = fetch_temperature(start_hist, end_hist)

                forecast_start = end_hist + pd.Timedelta(days=1)
                forecast_end = forecast_start + pd.Timedelta(days=180)
                st.session_state['df_fc'] = fetch_temperature_forecast(forecast_start, forecast_end)
            else:
                st.warning("Storage table empty — temperature data not fetched.")
        except Exception as e:
            st.error(f"Error fetching temperature data: {e}")

    st.success("Data fetched and cached!")

# Load from DB
def load_table(table_name):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY period ASC", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

if show_prices:
    df_prices = fetch_prices()
    st.subheader("Henry Hub Weekly Average Gas Prices")

    # compute weekly average prices (week ending Saturday)
    weekly_prices = df_prices.resample('W-SAT', on='period')['value'].mean().reset_index()
    weekly_prices = weekly_prices.sort_values('period').reset_index(drop=True)

    # summary table
    st.line_chart(weekly_prices.set_index("period")["value"])

    # --- Weekly averages for the ENTIRE year with percentile-based bands (last 5 years) ---
    max_date = weekly_prices['period'].max()
    current_year = max_date.year

    # Current year weekly series (weeks that fall in current year)
    cur_weekly = weekly_prices[weekly_prices['period'].dt.year == current_year].copy()
    cur_weekly['week'] = cur_weekly['period'].dt.isocalendar().week.astype(int)

    # Historical data: previous years (up to hist_years prior years) from weekly series
    hist = weekly_prices[(weekly_prices['period'].dt.year < current_year) &
                         (weekly_prices['period'].dt.year >= (current_year - hist_years))].copy()

    if hist.empty:
        st.info("Not enough historical weekly data (last 5 years) to compute percentile statistics.")
    else:
        hist['week'] = hist['period'].dt.isocalendar().week.astype(int)

        # percentile half-width slider: 0..50 (default 25 => lower=25th, upper=75th)
        half_width = st.slider("Percentile half-width (use 25 for 25th/75th)", 0, 50, 25, 1)
        low_pct = max(0, 50 - half_width) / 100.0
        mid_pct = 0.5
        high_pct = min(100, 50 + half_width) / 100.0

        # compute percentiles by week-of-year from historical weekly averages
        pct = hist.groupby('week')['value'].quantile([low_pct, mid_pct, high_pct]).unstack(level=1)
        pct.columns = ['p_low', 'p50', 'p_high']
        pct = pct.reset_index()

        # full-year weekly index (Saturdays)
        weeks = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31", freq='W-SAT')
        weeks_df = pd.DataFrame({'period': weeks})
        weeks_df['week'] = weeks_df['period'].dt.isocalendar().week.astype(int)

        # merge percentiles and current actuals (actual may be NaN for future weeks)
        weeks_df = weeks_df.merge(pct, on='week', how='left')
        weeks_df = weeks_df.merge(cur_weekly[['week', 'value']].rename(columns={'value': 'actual'}), on='week', how='left')

        # prepare display frame
        display_df = weeks_df.set_index('period')[['actual', 'p50', 'p_low', 'p_high']].sort_index()
        display_df = display_df.rename(columns={'p50': 'median', 'p_low': 'lower', 'p_high': 'upper'})

        st.subheader(f"{current_year}: Weekly average prices (actual) vs historical percentile bands (last 5 years)")
        st.line_chart(display_df)


if show_production:
    df_production = fetch_production()
    weekly_production = df_production.sort_values("period").reset_index(drop=True)
    max_date = weekly_production['period'].max()
    current_year = max_date.year

    cur_weekly = weekly_production[weekly_production['period'].dt.year == current_year].copy()
    cur_weekly['week'] = cur_weekly['period'].dt.isocalendar().week.astype(int)

    # Historical data: previous years (up to hist_years prior years)
    hist = weekly_production[(weekly_production['period'].dt.year < current_year) &
                          (weekly_production['period'].dt.year >= (current_year - hist_years))].copy()
    
    # Regression to forecast production
    withdrawals = df_production.set_index("period")['value'].copy()

    recent_months = st.slider("Months of recent data to use for production regression:", min_value=12, max_value=len(df_production), value=150, step=1)

    df_reg = pd.DataFrame(index=withdrawals.index)
    df_reg = df_reg.sort_index()
    df_reg['withdrawals'] = withdrawals
    df_reg['lag_1'] = withdrawals.shift(-1)
    df_reg['lag_12'] = withdrawals.shift(-12)
    df_reg['time_trend'] = np.arange(len(withdrawals))
    df_reg = df_reg.iloc[-recent_months:]

    # Month dummies
    df_months = pd.get_dummies(df_reg.index.month, prefix='month')
    df_months.index = df_reg.index
    df_reg = pd.concat([df_reg, df_months], axis=1)

    df_reg = df_reg.dropna()

    month_cols = [f'month_{i}' for i in range(1,13)]
    df_reg[month_cols] = df_reg[month_cols].astype(int)

    # -----------------------
    # Train
    # -----------------------
    X = df_reg.drop(columns=['withdrawals'])
    X = sm.add_constant(X)
    y = df_reg['withdrawals']

    # Fit model
    model = sm.OLS(y, X).fit()
    
    # Forecast
    forecast_months = 48

    # Last observed date
    last_date = df_reg.index[-1]

    # Create a monthly datetime index starting from the next month
    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), 
                                periods=forecast_months, 
                                freq='M')
    preds = []
    last_obs = df_reg.iloc[-1].copy()

    for i in range(forecast_months):
        new_row = {}
        new_row['const'] = 1
        new_row['lag_1'] = last_obs['withdrawals'] if i == 0 else preds[-1]
        if i < 12:
            new_row['lag_12'] = df_reg['withdrawals'].iloc[-12 + i]
        else:
            new_row['lag_12'] = preds[-12]
        new_row['time_trend'] = last_obs['time_trend'] + i + 1

        # Month dummies
        future_month = (last_obs.name.month + i + 1 - 1) % 12 + 1
        for m in range(1, 13):
            new_row[f'month_{m}'] = 1 if m == future_month else 0

        pred = model.predict(pd.DataFrame([new_row]))[0]
        preds.append(pred)

    forecast_series = pd.Series(preds, index=forecast_index, name='Forecast')

    st.subheader("US Monthly Gas Production & Regression Forecast 48 Months Ahead")

    # raw weekly series (storage is weekly already)
    st.line_chart(pd.concat([df_production.set_index("period")["value"], forecast_series], axis=1))
    st.subheader("Regression Summary")
    results_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std Err': model.bse,
        't': model.tvalues,
        'p-value': model.pvalues
    })
    st.dataframe(results_df)


if show_storage:
    df_storage = fetch_storage()
    st.subheader("US Weekly Gas Storage")

    # raw weekly series (storage is weekly already)
    st.line_chart(df_storage.set_index("period")["value"])

    # --- Weekly averages for the ENTIRE year with percentile-based bands (last 5 years) ---
    weekly_storage = df_storage.sort_values("period").reset_index(drop=True)
    max_date = weekly_storage['period'].max()
    current_year = max_date.year

    # Current year weekly series
    cur_weekly = weekly_storage[weekly_storage['period'].dt.year == current_year].copy()
    cur_weekly['week'] = cur_weekly['period'].dt.isocalendar().week.astype(int)

    # Historical data: previous years (up to hist_years prior years)
    hist = weekly_storage[(weekly_storage['period'].dt.year < current_year) &
                          (weekly_storage['period'].dt.year >= (current_year - hist_years))].copy()

    if hist.empty:
        st.info("Not enough historical weekly storage data (last 5 years) to compute percentile statistics.")
    else:
        hist['week'] = hist['period'].dt.isocalendar().week.astype(int)

        # percentile half-width slider: 0..50 (default 25 => lower=25th, upper=75th)
        half_width_storage = st.slider("Storage percentile half-width (use 25 for 25th/75th)", 0, 50, 25, 1)
        low_pct = max(0, 50 - half_width_storage) / 100.0
        mid_pct = 0.5
        high_pct = min(100, 50 + half_width_storage) / 100.0

        # compute percentiles by week-of-year from historical weekly storage
        pct = hist.groupby('week')['value'].quantile([low_pct, mid_pct, high_pct]).unstack(level=1)
        pct.columns = ['p_low', 'p50', 'p_high']
        pct = pct.reset_index()

        # full-year weekly index (Saturdays)
        weeks = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31", freq='W-SAT')
        weeks_df = pd.DataFrame({'period': weeks})
        weeks_df['week'] = weeks_df['period'].dt.isocalendar().week.astype(int)

        # merge percentiles and current actuals (actual may be NaN for future weeks)
        weeks_df = weeks_df.merge(pct, on='week', how='left')
        weeks_df = weeks_df.merge(cur_weekly[['week', 'value']].rename(columns={'value': 'actual'}), on='week', how='left')

        # prepare display frame and rename columns
        display_df = weeks_df.set_index('period')[['actual', 'p50', 'p_low', 'p_high']].sort_index()
        display_df = display_df.rename(columns={'p50': 'median', 'p_low': 'lower', 'p_high': 'upper'})

        st.subheader(f"{current_year}: Weekly storage (actual) vs historical percentile bands (last 5 years)")
        st.line_chart(display_df)

    
    st.subheader("Weekly Storage Change Actual vs Market Forecast")
    df = pd.read_excel("data/forecast_surprise.xlsx")
    
    # --- Data cleaning ---
    def to_number(x):
        if isinstance(x, str) and 'B' in x:
            return float(x.replace('B', '').replace('-', '-'))
        try:
            return float(x)
        except:
            return None

    for col in ["Actual", "Forecast", "Previous"]:
        df[col] = df[col].apply(to_number)

    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df = df.sort_values("Release Date")

    # --- Filter latest N ---
    n = st.slider("Select number of latest releases", 5, len(df), 10)
    df_recent = df.tail(n)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_recent["Release Date"],
        y=df_recent["Actual"],
        name="Actual",
        marker_color="royalblue",
        opacity=0.85,
        width=0.6
    ))

    fig.add_trace(go.Scatter(
        x=df_recent["Release Date"],
        y=df_recent["Forecast"],
        mode="markers",
        name="Forecast",
        marker=dict(color="orange", size=10, symbol="circle"),
        connectgaps=False  # ensures no connecting lines between points
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Release Date",
        yaxis_title="BCf",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Metric",
        hovermode="x unified"
    )
    print(df_recent)

    st.plotly_chart(fig, use_container_width=True)
    pred_eia_surprise_data = predict_storage_surprise()
    # Option 1: Display as a table
    release_date = pred_eia_surprise_data['Release Date'].strftime('%Y-%m-%d')
    direction = "Positive " if pred_eia_surprise_data['Pred_Direction'] == 1 else "Negative"
    probability = f"{pred_eia_surprise_data['Pred_Prob_Correct']:.2%}"
    st.markdown(
        f"""
        <div style="
            background-color:#0d1117; 
            padding:20px; 
            border-radius:15px; 
            color:#e6edf3; 
            text-align:center;
            border: 1px solid #1f6feb;
            box-shadow: 0 0 15px rgba(31, 111, 235, 0.2);
        ">
            <h2 style="color:#58a6ff;">EIA Storage Surprise Prediction Details</h2>
            <p style="font-size:18px;">
                <strong>Release Date:</strong> {release_date}
            </p>
            <p style="font-size:18px;">
                <strong>Predicted Direction:</strong> 
                <span style="color:{'lightgreen' if direction.startswith('Positive') else '#ff4d4d'};">
                    {direction}
                </span>
            </p>
            <p style="font-size:18px;">
                <strong>Implied Probability Predicted Direction is Correct:</strong> 
                <span style="color:#a5d6ff;">{probability}</span>
            </p>
        </div>
        """, unsafe_allow_html=True
    )



# ========================
# FORECAST STORAGE USING REGRESSION
# ========================
def forecast_storage_regression():
    df_storage = fetch_storage()
    if df_storage.empty:
        return {"percentiles": [], "scenarios": {}}

    # Historical temperature
    start_hist = df_storage['period'].min() - pd.Timedelta(days=7*5*52)
    end_hist = df_storage['period'].max()
    df_hist = fetch_temperature(start_hist, end_hist)

    # Aggregate temperature to weekly storage
    df_storage['week_start'] = df_storage['period'] - pd.to_timedelta(6, unit='d')
    weekly_hdd_cdd = []
    for _, row in df_storage.iterrows():
        mask = (df_hist['date'] >= row['week_start']) & (df_hist['date'] <= row['period'])
        weekly_hdd_cdd.append([df_hist.loc[mask, 'HDD'].sum(), df_hist.loc[mask, 'CDD'].sum()])
    weekly_hdd_cdd = np.array(weekly_hdd_cdd)
    df_storage['HDD'] = weekly_hdd_cdd[:, 0]
    df_storage['CDD'] = weekly_hdd_cdd[:, 1]

    # Delta & relative storage
    df_storage['deltaS'] = df_storage['value'].diff()
    df_storage = df_storage.dropna(subset=['deltaS']).reset_index(drop=True)
    df_storage['week'] = df_storage['period'].dt.isocalendar().week
    df_storage['year'] = df_storage['period'].dt.year
    df_storage['storage_5yr_mean'] = df_storage.groupby('week')['value'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df_storage['storage_rel'] = df_storage['value'] - df_storage['storage_5yr_mean']

    # Month dummies
    df_storage['month'] = df_storage['period'].dt.month
    month_dummies = pd.get_dummies(df_storage['month'], prefix='month', drop_first=True)
    df_storage = pd.concat([df_storage, month_dummies], axis=1)
    df_storage[month_dummies.columns] = df_storage[month_dummies.columns].astype(float)

    # Regression
    features = ['HDD','CDD','storage_rel'] + list(month_dummies.columns)
    df_storage[features] = df_storage[features].apply(pd.to_numeric, errors='coerce')
    df_storage = df_storage.dropna(subset=features + ['deltaS'])

    X = sm.add_constant(df_storage[features])
    y = df_storage['deltaS'].astype(float)
    model = sm.OLS(y, X).fit()

    # Forecast
    forecast_start = df_storage['period'].max() + pd.Timedelta(days=1)
    forecast_end = forecast_start + pd.Timedelta(days=180)
    df_fc = fetch_temperature_forecast(forecast_start, forecast_end)
    df_fc_weekly = df_fc.resample('W-SAT', on='date').sum().reset_index()
    df_fc_weekly['month'] = df_fc_weekly['date'].dt.month
    df_fc_weekly['storage_rel'] = df_storage['storage_rel'].iloc[-1]

    month_dummies_fc = pd.get_dummies(df_fc_weekly['month'], prefix='month', drop_first=True)
    df_fc_weekly = pd.concat([df_fc_weekly, month_dummies_fc], axis=1)
    for f in features:
        if f not in df_fc_weekly.columns:
            df_fc_weekly[f] = 0

    X_forecast = sm.add_constant(df_fc_weekly[features], has_constant='add')

    # Monte Carlo
    n_sims = 1000
    deltaS_sim = np.zeros((len(df_fc_weekly), n_sims))
    sigma_resid = np.std(model.resid)
    np.random.seed(42)
    for i in range(n_sims):
        noise = np.random.normal(0, sigma_resid, size=len(df_fc_weekly))
        deltaS_sim[:,i] = model.predict(X_forecast) + noise

    last_storage = df_storage['value'].iloc[-1]
    storage_sim = deltaS_sim.cumsum(axis=0) + last_storage

    storage_p10 = np.percentile(storage_sim,10,axis=1)
    storage_p50 = np.percentile(storage_sim,50,axis=1)
    storage_p90 = np.percentile(storage_sim,90,axis=1)

    return {
        "percentiles":[
            {"period":str(df_fc_weekly['date'].iloc[i]), "p10":float(storage_p10[i]),
             "p50":float(storage_p50[i]), "p90":float(storage_p90[i])}
            for i in range(len(df_fc_weekly))
        ]
    }

# Removed redundant early regression-trigger block to avoid double plotting.
# The regression forecast is now run and displayed once later under the "show_regression_forecast" handling.

# Button to trigger regression forecast (removed redundant button; regression runs when checkbox is checked)
if show_regression_forecast:
    with st.spinner("Forecasting storage..."):
        forecast = forecast_storage_regression()
    st.subheader("Storage Forecast (Regression)")
    df_fc_display = pd.DataFrame(forecast['percentiles'])
    df_fc_display['period'] = pd.to_datetime(df_fc_display['period'])
    st.line_chart(df_fc_display.set_index('period')[['p10','p50','p90']])
    st.dataframe(df_fc_display)

# === Forecast panels controlled by their own checkboxes ===
# These run even if the "Show Gas Storage" panel isn't visible.
if show_mc_forecast:
    st.subheader("Monte Carlo Storage Forecast (sample historical moves by week-of-year)")

    # MC controls (sidebar)
    mc_sims = st.slider("MC simulations", 100, 5000, 1000, step=100, key="mc_sims")
    mc_band = st.slider("MC central band percentile (e.g. 90 -> 5th/50th/95th)", 50, 99, 90, 1, key="mc_band")
    mc_hist_years = st.slider("MC: years of historical movements to sample", 1, 20, hist_years, 1, key="mc_hist_years")

    # load storage table (independent of show_storage)
    weekly_storage = fetch_storage().sort_values("period").reset_index(drop=True)
    if weekly_storage.empty:
        st.info("No storage data available. Run 'Fetch Latest Data' first.")
    else:
        # prepare historical deltas grouped by week-of-year (exclude current year)
        current_year_ws = weekly_storage['period'].max().year
        hist_mask_mc = (weekly_storage['period'].dt.year < current_year_ws) & \
                       (weekly_storage['period'].dt.year >= (current_year_ws - mc_hist_years))
        hist_for_mc = weekly_storage.loc[hist_mask_mc].copy()
        if hist_for_mc.empty:
            st.info("Not enough historical weekly storage rows for Monte Carlo. Expand the historical range.")
        else:
            hist_for_mc['year'] = hist_for_mc['period'].dt.year
            hist_for_mc['week'] = hist_for_mc['period'].dt.isocalendar().week.astype(int)
            hist_for_mc['deltaS'] = hist_for_mc.groupby('year')['value'].diff()
            deltas_by_week = (
                hist_for_mc.dropna(subset=['deltaS'])
                .groupby('week')['deltaS']
                .apply(lambda s: s.values)
                .to_dict()
            )
            all_deltas = hist_for_mc['deltaS'].dropna().values
            if len(all_deltas) == 0:
                st.info("No historical weekly deltas available for sampling.")
            else:
                last_obs = weekly_storage['period'].max()
                fc_periods = pd.date_range(start=last_obs + pd.Timedelta(days=7), periods=26, freq='W-SAT')
                fc_weeks = fc_periods.isocalendar().week.astype(int).to_numpy()

                rng = np.random.default_rng(42)
                sampled_deltas = np.zeros((len(fc_weeks), mc_sims), dtype=float)
                for i, wk in enumerate(fc_weeks):
                    pool = deltas_by_week.get(int(wk), None)
                    if pool is None or len(pool) == 0:
                        pool = all_deltas
                    sampled_deltas[i, :] = rng.choice(pool, size=mc_sims, replace=True)

                last_storage = weekly_storage['value'].iloc[-1]
                storage_sim = sampled_deltas.cumsum(axis=0) + last_storage

                lower_q = (100.0 - mc_band) / 2.0
                upper_q = 100.0 - lower_q
                qs = [lower_q, 50.0, upper_q]
                pct_vals = np.percentile(storage_sim, qs, axis=1)

                df_mc = pd.DataFrame({
                    "period": fc_periods,
                    "p_lower": pct_vals[0, :],
                    "p50": pct_vals[1, :],
                    "p_upper": pct_vals[2, :]
                }).set_index('period')

                st.line_chart(df_mc[['p_lower', 'p50', 'p_upper']])
                st.dataframe(df_mc.style.format("{:.2f}"))

# Monthly HDD/CDD this year vs historical percentiles (uses cached temperature in session_state)
if show_temp_stats:
     # require temperature history loaded by "Fetch Latest Data"
    if 'df_hist' not in st.session_state:
        st.info("Temperature history not loaded. Press 'Fetch Latest Data' to load temperatures for monthly HDD/CDD charts.")
    else:
        df_temp = st.session_state['df_hist'].copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month

        current_year = df_temp['year'].max()

        # Aggregate HDD/CDD by year-month
        ym = df_temp.groupby(['year','month'])[['HDD','CDD']].sum().reset_index()

        # Current year monthly HDD/CDD (may be partial for current month)
        cur_monthly = ym[ym['year'] == current_year].set_index('month')[['HDD','CDD']]

        # Historical years range using hist_years slider
        hist_mask = (ym['year'] < current_year) & (ym['year'] >= (current_year - hist_years))
        hist = ym[hist_mask].copy()
        if hist.empty:
            st.info(f"Not enough historical temperature data for the last {hist_years} years to compute percentiles.")
        else:
            # percentile half-width slider for HDD/CDD percentiles (applies to both)
            pct_half = st.slider("Temperature percentile half-width (e.g. 25 => 25th/50th/75th)", 0, 50, 25, 1)
            low_q = max(0, 50 - pct_half) / 100.0
            mid_q = 0.5
            high_q = min(100, 50 + pct_half) / 100.0

            # compute percentiles by month across historical years using chosen quantiles
            def pct_df(df, col):
                p = df.groupby('month')[col].quantile([low_q, mid_q, high_q]).unstack(level=1)
                p.columns = ['p_low','p50','p_high']
                return p

            h_hdd = pct_df(hist, 'HDD')
            h_cdd = pct_df(hist, 'CDD')

            # build full-year month index (1..12) with date index at first of month
            months = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-01", freq='MS')
            months_df = pd.DataFrame({'period': months, 'month': months.month})

            # merge current and historical percentiles
            months_df = months_df.merge(cur_monthly.reset_index(), on='month', how='left')
            months_df = months_df.merge(h_hdd.reset_index(), on='month', how='left')
            months_df = months_df.merge(h_cdd.reset_index(), on='month', how='left', suffixes=('','_cdd'))
            months_df = months_df.set_index('period')
            # ensure columns named consistently
            months_df.rename(columns={'HDD':'cur_HDD','CDD':'cur_CDD'}, inplace=True)

            # If forecast temperature present, aggregate monthly HDD/CDD and merge as fc_HDD/fc_CDD
            if 'df_fc' in st.session_state:
                df_fc = st.session_state['df_fc'].copy()
                df_fc['date'] = pd.to_datetime(df_fc['date'])
                df_fc['month'] = df_fc['date'].dt.month
                fc_monthly = df_fc.groupby('month')[['HDD','CDD']].sum().reset_index().rename(
                    columns={'HDD':'fc_HDD','CDD':'fc_CDD'}
                )

                # Don't include forecast for the current (most recent historical) month
                last_hist_month = df_temp['date'].max().month
                fc_monthly.loc[fc_monthly['month'] == last_hist_month, ['fc_HDD','fc_CDD']] = np.nan

                months_df = months_df.reset_index().merge(fc_monthly, on='month', how='left').set_index('period')
            else:
                months_df = months_df  # no forecast available

            # Chart HDD (include forecast line if available)
            st.subheader(f"{current_year}: Monthly HDD — actual vs historical percentiles (last {hist_years} yrs)")
            hdd_cols = ['cur_HDD','p_low','p50','p_high']
            if 'fc_HDD' in months_df.columns:
                hdd_cols.append('fc_HDD')
            hdd_display = months_df[hdd_cols].sort_index()
            st.line_chart(hdd_display)

            # Chart CDD (include forecast line if available)
            st.subheader(f"{current_year}: Monthly CDD — actual vs historical percentiles (last {hist_years} yrs)")
            cdd_cols = ['cur_CDD','p_low_cdd','p50_cdd','p_high_cdd']
            if 'fc_CDD' in months_df.columns:
                cdd_cols.append('fc_CDD')
            cdd_display = months_df[cdd_cols].rename(
                columns={'p_low_cdd':'p_low','p50_cdd':'p50','p_high_cdd':'p_high'}
            ).sort_index()
            st.line_chart(cdd_display)


# Top-level: Percentile ratio view (storage vs price) — runs whenever its checkbox is checked
if show_percentile_ratio:
    # separate historical-window slider for the percentile-ratio view (used to build comparison pools)
    ratio_hist_years = st.slider(
        "Percentile-ratio historical years (for weekly percentiles)",
        1, 20, hist_years, 1, key="ratio_hist_years"
    )
    # number of years to plot (including the most recent year)
    ratio_plot_years = st.slider(
        "Years to plot for percentile-ratio (including most recent)",
        1, 20, 5, 1, key="ratio_plot_years"
    )

    ws = fetch_storage().sort_values("period").reset_index(drop=True)
    pf = fetch_prices().sort_values("period").reset_index(drop=True)

    if ws.empty and pf.empty:
        st.info("No storage or price data available. Run 'Fetch Latest Data' first.")
    else:
        # weekly aggregates (week ending Saturday)
        if not ws.empty:
            ws_weekly = ws.resample('W-SAT', on='period')['value'].mean().reset_index()
            ws_weekly['year'] = ws_weekly['period'].dt.year
            ws_weekly['week'] = ws_weekly['period'].dt.isocalendar().week.astype(int)
        else:
            ws_weekly = pd.DataFrame(columns=['period','value','year','week'])

        if not pf.empty:
            pr_weekly = pf.resample('W-SAT', on='period')['value'].mean().reset_index()
            pr_weekly['year'] = pr_weekly['period'].dt.year
            pr_weekly['week'] = pr_weekly['period'].dt.isocalendar().week.astype(int)
        else:
            pr_weekly = pd.DataFrame(columns=['period','value','year','week'])

        # determine latest available week across both series
        latest_ws = ws_weekly['period'].max() if not ws_weekly.empty else pd.NaT
        latest_pr = pr_weekly['period'].max() if not pr_weekly.empty else pd.NaT
        latest = max(latest_ws, latest_pr, pd.Timestamp(datetime.now()))
        latest = pd.to_datetime(latest)

        # build weekly index covering the last `ratio_plot_years` years up to latest
        start_year = latest.year - ratio_plot_years + 1
        start_date = pd.Timestamp(f"{start_year}-01-01")
        weeks = pd.date_range(start=start_date, end=latest, freq='W-SAT')

        rows = []
        for d in weeks:
            wk = int(d.isocalendar().week)
            yr = int(d.year)

            # current-week values (may be NaN if missing)
            stor_cur_series = ws_weekly.loc[ws_weekly['period'] == d, 'value']
            stor_cur = float(stor_cur_series.iloc[0]) if len(stor_cur_series) > 0 else np.nan

            pr_cur_series = pr_weekly.loc[pr_weekly['period'] == d, 'value']
            pr_cur = float(pr_cur_series.iloc[0]) if len(pr_cur_series) > 0 else np.nan

            # historical pools for this week-of-year relative to this point's year
            stor_hist = ws_weekly[
                (ws_weekly['year'] < yr) &
                (ws_weekly['year'] >= (yr - ratio_hist_years)) &
                (ws_weekly['week'] == wk)
            ]['value'].values

            pr_hist = pr_weekly[
                (pr_weekly['year'] < yr) &
                (pr_weekly['year'] >= (yr - ratio_hist_years)) &
                (pr_weekly['week'] == wk)
            ]['value'].values

            # percentiles and ratios (percentile - 50)
            stor_pct = percentile_rank(stor_hist, stor_cur) if not np.isnan(stor_cur) else np.nan
            pr_pct = percentile_rank(pr_hist, pr_cur) if not np.isnan(pr_cur) else np.nan
            stor_ratio = stor_pct - 50.0 if not np.isnan(stor_pct) else np.nan
            pr_ratio = pr_pct - 50.0 if not np.isnan(pr_pct) else np.nan

            # sum of ratios (if one is missing use the other; if both missing -> NaN)
            if np.isnan(stor_ratio) and np.isnan(pr_ratio):
                sum_ratio = np.nan
            elif np.isnan(stor_ratio):
                sum_ratio = pr_ratio
            elif np.isnan(pr_ratio):
                sum_ratio = stor_ratio
            else:
                sum_ratio = stor_ratio + pr_ratio

            rows.append({
                "period": d,
                "year": yr,
                "week": wk,
                "storage_ratio": stor_ratio,
                "price_ratio": pr_ratio,
                "sum_ratio": sum_ratio,
                "storage_pct": stor_pct,
                "price_pct": pr_pct,
                "storage_cur": stor_cur,
                "price_cur": pr_cur
            })

        df_ratio = pd.DataFrame(rows).set_index('period')

        if df_ratio[['storage_ratio','price_ratio']].dropna(how='all').empty:
            st.info("No weekly values available in the selected plot range to compute percentile ratios.")
        else:
            st.subheader(f"Weekly percentile-ratio (percentile - 50) — plotting last {ratio_plot_years} years, comparing to prior {ratio_hist_years} yrs")
            st.line_chart(df_ratio[['storage_ratio','price_ratio','sum_ratio']])
            st.markdown("Table: weekly percentile ratios, year/week and current values")
            st.dataframe(df_ratio[['year','week','storage_ratio','price_ratio','sum_ratio','storage_pct','price_pct','storage_cur','price_cur']].style.format({
                "storage_ratio": "{:.1f}",
                "price_ratio": "{:.1f}",
                "sum_ratio": "{:.1f}",
                "storage_pct": "{:.1f}",
                "price_pct": "{:.1f}",
                "storage_cur": "{:.2f}",
                "price_cur": "{:.2f}"
            }))

# --- New: monthly-percentile bins and expected next-week return lookup (with separate sliders) ---
def compute_weekly_bins_and_returns(percentile_years=5):
    """
    Build weekly merged frame with monthly percentiles computed using `percentile_years`
    (for each calendar month we compute percentiles across that month's values in the
    percentile_years lookback window).
    """
    df_p = fetch_prices()
    df_s = fetch_storage()
    if df_p.empty or df_s.empty:
        return None

    weekly_prices = df_p.resample('W-SAT', on='period')['value'].mean().reset_index().rename(columns={'value':'price'})
    weekly_storage = df_s.resample('W-SAT', on='period')['value'].mean().reset_index().rename(columns={'value':'storage'})

    df = pd.merge(weekly_prices, weekly_storage, on='period', how='inner').sort_values('period').reset_index(drop=True)
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    df['week'] = df['period'].dt.isocalendar().week.astype(int)

    # next-week return (pct)
    df['price_next'] = df['price'].shift(-1)
    df['next_ret_pct'] = (df['price_next'] - df['price']) / df['price'] * 100.0

    latest_year = int(df['year'].max())
    start_year = latest_year - int(percentile_years) + 1

    # month-by-month percentile pools (over percentile_years)
    df['price_pct'] = np.nan
    df['stor_pct'] = np.nan
    for m in range(1, 13):
        pool = df[(df['month'] == m) & (df['year'] >= start_year) & (df['year'] <= latest_year)]
        pool_price_vals = pool['price'].dropna().values
        pool_stor_vals = pool['storage'].dropna().values
        if pool_price_vals.size == 0 or pool_stor_vals.size == 0:
            continue
        mask = df['month'] == m
        df.loc[mask, 'price_pct'] = df.loc[mask, 'price'].apply(lambda v: percentile_rank(pool_price_vals, v))
        df.loc[mask, 'stor_pct']  = df.loc[mask, 'storage'].apply(lambda v: percentile_rank(pool_stor_vals, v))

    # bins: tertiles on percentile (fixed cutoffs)
    def pct_to_bin(p):
        if np.isnan(p):
            return "unknown"
        if p <= 33.3333:
            return "low"
        if p <= 66.6667:
            return "mid"
        return "high"

    df['price_bin'] = df['price_pct'].apply(pct_to_bin)
    df['stor_bin']  = df['stor_pct'].apply(pct_to_bin)

    return df

# UI: select month and bins, show expected return for matching historical weeks
show_bin_forecast = st.sidebar.checkbox("Bin-based Monthly Expected Return")
if show_bin_forecast:
    # slider that controls how many years are used to compute the monthly percentiles
    percentile_lookback_years = st.slider(
        "Years to use for percentile calculation (per-month pools)",
        1, 20, hist_years, 1, key="percentile_lookback_years"
    )

    # slider that controls how many years of historical weeks are searched for matching samples
    data_lookback_years = st.slider(
        "Years of historical data to use when computing expected returns",
        1, 20, hist_years, 1, key="data_lookback_years"
    )

    # build dataframe with percentiles computed using percentile_lookback_years
    df_bins = compute_weekly_bins_and_returns(percentile_years=percentile_lookback_years)
    if df_bins is None:
        st.info("Need price and storage data — run 'Fetch Latest Data' first.")
    else:
        # show current (most recent) weekly percentiles and bins
        try:
            latest_period = df_bins['period'].max()
            latest_row = df_bins[df_bins['period'] == latest_period].iloc[-1]
            cur_price_pct = latest_row.get('price_pct', np.nan)
            cur_stor_pct = latest_row.get('stor_pct', np.nan)
            cur_price_bin = latest_row.get('price_bin', "unknown")
            cur_stor_bin = latest_row.get('stor_bin', "unknown")
            st.subheader("Most recent weekly percentiles & bins")
            st.markdown(
                f"- Date: {pd.to_datetime(latest_period).date()}  \n"
                f"- Price percentile: {cur_price_pct:.1f} → bin: **{cur_price_bin}**  \n"
                f"- Storage percentile: {cur_stor_pct:.1f} → bin: **{cur_stor_bin}**"
            )
        except Exception:
            st.info("Most recent percentiles/bins not available.")

        # controls
        month_opt = [(m, calendar.month_name[m]) for m in range(1,13)]
        default_idx = int(df_bins['month'].max())-1 if not df_bins.empty else 0
        sel = st.selectbox("Select calendar month", options=month_opt, index=default_idx,
                           format_func=lambda x: f"{x[0]:02d} - {x[1]}")
        target_month = sel[0]

        price_bin = st.selectbox("Price ratio bin", options=["low","mid","high"], index=1)
        stor_bin  = st.selectbox("Storage ratio bin", options=["low","mid","high"], index=1)

        # filter historical rows that match month and bins and have next-week return available
        latest_year = int(df_bins['year'].max())
        start_year_data = latest_year - int(data_lookback_years) + 1
        pool = df_bins[
            (df_bins['month'] == target_month) &
            (df_bins['price_bin'] == price_bin) &
            (df_bins['stor_bin'] == stor_bin) &
            (df_bins['year'] >= start_year_data) &
            (df_bins['year'] <= latest_year)
        ].copy()

        pool = pool.dropna(subset=['next_ret_pct'])
        n = len(pool)
        if n == 0:
            st.info(f"No matching historical weeks for {calendar.month_name[target_month]} / price={price_bin} / storage={stor_bin} in last {data_lookback_years} years.")
        else:
            mean_ret = pool['next_ret_pct'].mean()
            median_ret = pool['next_ret_pct'].median()
            p10 = np.percentile(pool['next_ret_pct'], 10)
            p90 = np.percentile(pool['next_ret_pct'], 90)

            st.subheader("Expected next-week return for matching historical weeks")
            st.write(f"Month: {calendar.month_name[target_month]}, Price bin: {price_bin}, Storage bin: {stor_bin}")
            st.write(f"Samples (years used for matching): {n}")
            st.write(f"Mean next-week return: {mean_ret:.3f}%")
            st.write(f"Median next-week return: {median_ret:.3f}%")
            st.write(f"10th pct: {p10:.3f}%, 90th pct: {p90:.3f}%")

            st.markdown("Distribution of historical next-week returns for matching weeks (by date):")
            df_bar = pool.dropna(subset=['next_ret_pct']).set_index('period').sort_index()[['next_ret_pct']]
            if df_bar.empty:
                st.info("No return data to plot.")
            else:
                # use string dates as categorical x-axis to avoid big gaps for missing weeks
                df_bar = df_bar.reset_index()
                df_bar['period_str'] = df_bar['period'].dt.strftime('%Y-%m-%d')
                df_bar = df_bar.set_index('period_str')[['next_ret_pct']]
                st.bar_chart(df_bar)
            st.dataframe(pool[['period','year','week','price','storage','price_pct','stor_pct','next_ret_pct']].sort_values('period', ascending=False).head(200).style.format({
                "price":"{:.2f}","storage":"{:.1f}","price_pct":"{:.1f}","stor_pct":"{:.1f}","next_ret_pct":"{:.3f}"
            }))