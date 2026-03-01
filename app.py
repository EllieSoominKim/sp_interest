import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis import df, X, Y, models
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

st.set_page_config(page_title="S&P500 Dashboard", layout="wide")
st.title("S&P500 Return & Volatility Analysis Dashboard")

# -----코로나 이전 / 이후 분할-----
covid = pd.to_datetime("2020-03-01")
before = df[df.index < covid]
after = df[df.index >= covid]

split_date = covid
X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]
Y_train = Y[Y.index < split_date]
Y_test = Y[Y.index >= split_date]

# -----OLS 회귀-----
X_ols = sm.add_constant(df["rate_change"])
Y_ols = df["return"]
model_ols = sm.OLS(Y_ols, X_ols).fit()
preds_ols = model_ols.predict(X_ols)

# -----Lag Correlation-----
df_monthly = df[["rate_change","Rate","volatility","return"]].resample("ME").mean()

for lag in range(1,13):
    df_monthly[f"lag{lag}"] = df_monthly["rate_change"].shift(lag)

df_monthly.dropna(inplace=True)

lags = range(1,13)
corrs = [df_monthly["return"].corr(df_monthly["rate_change"].shift(l)) for l in lags]

# -----Model MAE 계산-----
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

# -----Subplot 생성-----
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "Correlation Heatmap", "Lag Correlation",
        "OLS Regression", "Volatility",
        "COVID Return", "COVID Volatility",
        "Model MAE", "Model Predictions"
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# -----Correlation Heatmap (1,1)-----
corr = df[["return","Rate","rate_change"]].corr()

fig.add_trace(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=corr.round(4).values,
        texttemplate="%{text}",
        showscale=False
    ),
    row=1, col=1
)

# -----Lag Correlation (1,2)-----
fig.add_trace(
    go.Scatter(
        x=list(lags),
        y=corrs,
        mode="lines+markers",
        line=dict(color="blue"),
        name="Lag Correlation"
    ),
    row=1, col=2
)

# -----OLS Regression (2,1)-----
fig.add_trace(
    go.Scatter(
        x=df["rate_change"],
        y=df["return"],
        mode="markers",
        marker=dict(size=4, opacity=0.3, color="blue"),
        name="Actual Return"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["rate_change"],
        y=preds_ols,
        mode="lines",
        line=dict(color="red"),
        name="OLS Prediction"
    ),
    row=2, col=1
)

# -----Volatility (2,2)-----
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["volatility"],
        mode="lines",
        line=dict(color="orange"),
        name="Volatility"
    ),
    row=2, col=2
)

# -----COVID Return (3,1)-----
fig.add_trace(
    go.Scatter(x=before.index, y=before["return"], mode="lines", name="Before COVID"),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=after.index, y=after["return"], mode="lines", name="After COVID"),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=[split_date, split_date],
        y=[df["return"].min(), df["return"].max()],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="COVID Split"
    ),
    row=3, col=1
)

# -----COVID Volatility (3,2)-----
fig.add_trace(
    go.Scatter(x=before.index, y=before["volatility"], mode="lines", name="Before COVID"),
    row=3, col=2
)

fig.add_trace(
    go.Scatter(x=after.index, y=after["volatility"], mode="lines", name="After COVID"),
    row=3, col=2
)

fig.add_trace(
    go.Scatter(
        x=[split_date, split_date],
        y=[df["volatility"].min(), df["volatility"].max()],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="COVID Split"
    ),
    row=3, col=2
)

# -----Model MAE (4,1)-----
models_list = list(avg_errors.keys())
before_vals = [avg_errors[m]["Before_COVID"] for m in models_list]
after_vals  = [avg_errors[m]["After_COVID"] for m in models_list]

fig.add_trace(
    go.Bar(x=models_list, y=before_vals, name="Before COVID"),
    row=4, col=1
)

fig.add_trace(
    go.Bar(x=models_list, y=after_vals, name="After COVID"),
    row=4, col=1
)

# -----Model Predictions (4,2)-----
for name, model in models.items():
    preds_after = model.predict(X_test)
    color = "purple" if name=="Random Forest" else "green"

    fig.add_trace(
        go.Scatter(
            x=Y_test.index,
            y=preds_after,
            mode="lines",
            line=dict(dash="dash", color=color),
            name=f"{name} Prediction"
        ),
        row=4, col=2
    )

fig.add_trace(
    go.Scatter(
        x=Y_test.index,
        y=Y_test,
        mode="lines",
        name="Actual Return (After COVID)"
    ),
    row=4, col=2
)

fig.add_trace(
    go.Scatter(
        x=[split_date, split_date],
        y=[Y_test.min(), Y_test.max()],
        mode="lines",
        line=dict(dash="dot", color="black"),
        name="COVID Split"
    ),
    row=4, col=2
)

# -----Layout-----
fig.update_layout(
    height=1200,
    width=1000,
    showlegend=True,
    title_text="<Integrated Financial Analysis Dashboard>",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)