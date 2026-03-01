import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_pipeline import load_data
import seaborn as sns

# -----데이터 불러오기-----
df = load_data()

# -----1. 상관계수 분석-----
corr = df[["return","Rate","rate_change"]].corr()
print("상관계수\n", corr, "\n")

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".4f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----월 단위 데이터 생성 + Lag 변수-----

df_monthly = df[["rate_change","Rate","volatility","return"]].resample("ME").mean()

for lag in range(1,13):
    df_monthly[f"lag{lag}"] = df_monthly["rate_change"].shift(lag)

df_monthly.dropna(inplace=True)


# -----2. Lag 상관관계 분석-----
lags = range(1,13)
corrs = [df_monthly["return"].corr(df_monthly["rate_change"].shift(l)) for l in lags]

plt.figure(figsize=(10,5))
plt.plot(lags,corrs,marker="o")
plt.title("Lag Correlation: Rate change vs Monthly Return")
plt.xlabel("Lag (Months)")
plt.ylabel("Correlation")
plt.grid(True)
plt.show()

# -----3. OLS 회귀 분석-----
X_ols = sm.add_constant(df["rate_change"])
Y_ols = df["return"]

model_ols = sm.OLS(Y_ols, X_ols).fit()
print(model_ols.summary())

preds_ols = model_ols.predict(X_ols)

plt.figure(figsize=(12,5))
plt.scatter(df["rate_change"], df["return"], alpha=0.3, label="Actual Return")
plt.plot(df["rate_change"], preds_ols, color="red", linewidth=2, label="OLS Prediction")
plt.title("OLS Regression: Return vs Rate change")
plt.xlabel("Rate change")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.show()

# -----4. 롤링 변동성 시계열-----
plt.figure(figsize=(12,4))
plt.plot(df.index, df["volatility"])
plt.title("S&P500 Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()

# -----5. 코로나 전후 비교-----

covid = pd.to_datetime("2020-03-01")

before = df[df.index < covid]
after = df[df.index >= covid]

print("===== 코로나 전후 통계 =====")

print("\nBefore COVID")
print("Return mean:", before["return"].mean())
print("Volatility mean:", before["volatility"].mean())

print("\nAfter COVID")
print("Return mean:", after["return"].mean())
print("Volatility mean:", after["volatility"].mean())


# 수익률 비교
plt.figure(figsize=(12,4))
plt.plot(before.index, before["return"], label="Before")
plt.plot(after.index, after["return"], label="After")
plt.axvline(covid, linestyle="--")
plt.title("S&P500 Monthly Return: Before vs After COVID")
plt.legend()
plt.grid(True)
plt.show()

# 변동성 비교
plt.figure(figsize=(12,4))
plt.plot(before.index, before["volatility"], label="Before")
plt.plot(after.index, after["volatility"], label="After")
plt.axvline(covid, linestyle="--")
plt.title("S&P500 Monthly Volatility: Before vs After COVID")
plt.legend()
plt.grid(True)
plt.show()

# -----6. 모델 비교 (선형회귀 vs 랜덤포레스트)-----
Y = df_monthly["return"]
X = df_monthly.drop(columns=["return"])

split_date = pd.to_datetime("2020-03-01")

X_train = X[X.index < split_date]
X_test  = X[X.index >= split_date]
Y_train = Y[Y.index < split_date]
Y_test  = Y[Y.index >= split_date]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

avg_errors = {}

for name, model in models.items():

    model.fit(X_train, Y_train)

    preds_before = model.predict(X_train)
    mae_before = mean_absolute_error(Y_train, preds_before)

    preds_after = model.predict(X_test)
    mae_after = mean_absolute_error(Y_test, preds_after)

    avg_errors[name] = {
        "Before_COVID": mae_before,
        "After_COVID": mae_after
    }

results_df = pd.DataFrame(avg_errors).T
print("=== 모델별 MAE ===")
print(results_df)


# 오차 그래프
avg_error_df = pd.DataFrame.from_dict(avg_errors, orient="index")
avg_error_df.plot(kind="bar", figsize=(10,6))
plt.title("Average Absolute Prediction Error")
plt.ylabel("MAE")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# 예측 비교 시각화
plt.figure(figsize=(14,6))

plt.plot(Y_train.index, Y_train, label="Actual (Before COVID)", linewidth=2)
plt.plot(Y_test.index, Y_test, label="Actual (After COVID)", linewidth=2)

for name, model in models.items():
    preds_after = model.predict(X_test)
    plt.plot(Y_test.index, preds_after, "--", label=f"{name} Prediction")

plt.axvline(split_date, linestyle=":", color="black", label="COVID Split")

plt.legend()
plt.title("Model Prediction Comparison")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.show()