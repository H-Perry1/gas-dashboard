import pandas as pd
import plotly.express as px
from meteostat import Daily, Stations
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression

def predict_storage_surprise():
    df_merged = pd.read_excel("data/forecast_surprise.xlsx")
    # Remove 'B' and convert to numeric
    for col in ['Actual', 'Forecast', 'Previous']:
        df_merged[col] = df_merged[col].astype(str).str.replace('B', '', regex=False)
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    # Convert release date to datetime
    df_merged['Release Date'] = pd.to_datetime(df_merged['Release Date'], format='%b %d, %Y')

    # Optional: compute surprise (Actual - Forecast)
    df_merged['Surprise'] = df_merged['Actual'] - df_merged['Forecast']

    df_merged = df_merged.sort_values('Release Date').reset_index(drop=True)
    df_merged['Surprise'] = df_merged['Actual'] - df_merged['Forecast']
    df_merged['Lagged_Surprise'] = df_merged['Surprise'].shift(1)
    df_merged['Lagged_Storage'] = df_merged['Actual'].shift(1)

    threshold = df_merged['Surprise'].quantile(0.75)
    df_merged['Extreme_Surprise'] = (df_merged['Surprise'].abs() > threshold).astype(int)
    df_merged['Extreme_Surprise_Roll'] = df_merged['Extreme_Surprise'].rolling(5).sum().shift(1)

    df_merged = df_merged.sort_values('Release Date')

    cities = {
        'Chicago': '72534',
        'New_York': '72503',  # replace with actual station ID
        'Houston': '72243'    # replace with actual station ID
    }

    start = pd.Timestamp.today() - pd.Timedelta(days=9000)  # 7 days ago
    end = pd.Timestamp.today()  # today

    threshold_C = 18.3

    for city, station_id in cities.items():
        # Fetch daily weather
        data = Daily(station_id, start, end).fetch()

        # Compute tavg if missing
        if 'tavg' not in data.columns or data['tavg'].isnull().all():
            data['tavg'] = data[['tmin', 'tmax']].mean(axis=1)

        # Compute HDD/CDD
        data['HDD'] = (threshold_C - data['tavg']).clip(lower=0)
        data['CDD'] = (data['tavg'] - threshold_C).clip(lower=0)
        data = data.reset_index().rename(columns={'time': 'Date'})

        # Compute weekly sums between release dates
        hdd_list, cdd_list = [], []
        release_dates = df_merged['Release Date'].tolist()
        for i in range(len(release_dates)):
            start_date = release_dates[i-1] + pd.Timedelta(days=1) if i > 0 else release_dates[i] - pd.Timedelta(days=7)
            end_date = min(release_dates[i], pd.Timestamp.today())
            mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
            hdd_list.append(data.loc[mask, 'HDD'].sum())
            cdd_list.append(data.loc[mask, 'CDD'].sum())

        df_merged[f'HDD_{city}'] = hdd_list
        df_merged[f'CDD_{city}'] = cdd_list

    # ===============================
    # 5. Add 5 lags for all cities + storage/surprise
    # ===============================
    lags = 3
    for city in cities.keys():
        for lag in range(1, lags + 1):
            df_merged[f'HDD_{city}_lag{lag}'] = df_merged[f'HDD_{city}'].shift(lag)
            df_merged[f'CDD_{city}_lag{lag}'] = df_merged[f'CDD_{city}'].shift(lag)

    for lag in range(1, lags + 1):
        df_merged[f'Lagged_Storage_lag{lag}'] = df_merged['Lagged_Storage'].shift(lag)
        df_merged[f'Lagged_Surprise_lag{lag}'] = df_merged['Surprise'].shift(lag)


    df_merged['Actual_MovingSum'] = (
        df_merged['Actual']
        .rolling(window=4, min_periods=1)
        .sum()
        .shift(1)
    )


    cities = ['Chicago', 'New_York', 'Houston']


    # Step 2: Build features
    X = df_merged.dropna()[
        [f'{var}_{city}_lag{lag}' for city in cities for var in ['HDD', 'CDD'] for lag in range(1, lags+1)]
        + [f'{var}_lag{lag}' for var in ['Lagged_Storage'] for lag in range(1, lags+1)]
        + [f'{var}_{city}' for city in cities for var in ['HDD', 'CDD']]
        + ['Actual_MovingSum', 'Extreme_Surprise_Roll']
    ]

    # Drop unnecessary columns if still needed
    X = X.drop(columns=['HDD_New_York', 'CDD_New_York', 'HDD_Houston', 'CDD_Houston','HDD_Chicago', 'CDD_Chicago'], errors='ignore')

    # Fill missing values
    X = X.fillna(method='ffill')

    # Generate 3rd-degree polynomial features (you currently have degree=1, change if needed)
    poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)

    # Column names for clarity
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

    # Add constant term for intercept
    X_poly = sm.add_constant(X_poly)

    # Define dependent variable
    y = df_merged.dropna()['Surprise']

    # Fit OLS regression
    model_poly = sm.OLS(y, X_poly).fit()

    # ===============================
    # 1. Build features exactly as before
    # ===============================
    cities = ['Chicago', 'New_York', 'Houston']

    X_all = df_merged[
        [f'{var}_{city}_lag{lag}' for city in cities for var in ['HDD', 'CDD'] for lag in range(1, lags+1)]
        + [f'{var}_lag{lag}' for var in ['Lagged_Storage'] for lag in range(1, lags+1)]
        + [f'{var}_{city}' for city in cities for var in ['HDD', 'CDD']]
        + ['Actual_MovingSum', 'Extreme_Surprise_Roll'] # current week
    ].dropna()

    # Fill missing values if any
    X_all = X_all.fillna(method='ffill')
    X_all = X_all.drop(columns=['HDD_New_York', 'CDD_New_York', 'HDD_Houston', 'CDD_Houston','HDD_Chicago', 'CDD_Chicago'])

    # ===============================
    # 2. Transform with PolynomialFeatures (same as during fitting)
    # ===============================
    X_all_poly = poly.transform(X_all)
    X_all_poly = pd.DataFrame(X_all_poly, columns=feature_names, index=X_all.index)

    # Add constant
    X_all_poly = sm.add_constant(X_all_poly)

    # ===============================
    # 3. Predict for all rows
    # ===============================
    df_merged['Predicted_Surprise'] = model_poly.predict(X_all_poly)

    # ===============================
    # 4. Preview
    # ===============================

    # ===============================
    # 3b. Get 95% prediction interval
    # ===============================
    pred = model_poly.get_prediction(X_all_poly)
    pred_summary = pred.summary_frame(alpha=0.2)  # 95% confidence

    # Add to df_merged
    df_merged['Predicted_Surprise'] = pred_summary['mean']
    df_merged['Predicted_Surprise_Lower'] = pred_summary['mean_ci_lower']
    df_merged['Predicted_Surprise_Upper'] = pred_summary['mean_ci_upper']

    # Add Significant column
    df_merged['Significant'] = ((df_merged['Predicted_Surprise_Lower'] > 0) & 
                                (df_merged['Predicted_Surprise_Upper'] > 0)) | \
                            ((df_merged['Predicted_Surprise_Lower'] < 0) & 
                                (df_merged['Predicted_Surprise_Upper'] < 0))

    # Convert boolean to integer
    df_merged['Significant'] = df_merged['Significant'].astype(int)

    # -------------------------------
    # 1. Prepare dataset
    # -------------------------------
    df = df_merged.copy()
    df = df.dropna(subset=['Predicted_Surprise'])

    df['Surprise_Dir'] = (df['Surprise'] > 0).astype(int)

    # Features
    X = df[['Predicted_Surprise', 'Predicted_Surprise_Lower', 'Predicted_Surprise_Upper']]

    # -------------------------------
    # 2. Train logistic regression on full dataset
    # -------------------------------
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_reg.fit(X, df['Surprise_Dir'])

    # -------------------------------
    # 3. Predicted direction and probability
    # -------------------------------
    pred_prob = log_reg.predict_proba(X)[:, 1]        # probability that Surprise_Dir = 1
    pred_dir = (pred_prob >= 0.5).astype(int)        # predicted direction (0=negative, 1=positive)

    # -------------------------------
    # 4. Compute probability predicted direction is correct
    # -------------------------------
    df['Pred_Direction'] = pred_dir
    df['Pred_Prob_Correct'] = np.where(pred_dir == 1, pred_prob, 1 - pred_prob)

    # -------------------------------
    # 5. Check results
    # -------------------------------
    print(df[['Surprise_Dir', 'Pred_Direction', 'Pred_Prob_Correct', 'Predicted_Surprise']].head())

    return df[['Release Date', 'Pred_Direction', 'Pred_Prob_Correct']].iloc[-1]