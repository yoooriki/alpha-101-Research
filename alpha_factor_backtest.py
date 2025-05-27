import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 获取多只美股数据
# 用10只美股
# 你可以根据需要替换成其他股票代码
# 例如：AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA, NFLX, INTC, IBM
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "INTC", "IBM"]
dfs = []
for t in tickers:
    df = ak.stock_us_daily(symbol=t)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.loc['2022-01-01':'2022-12-31']
    df['ticker'] = t
    dfs.append(df)
data = pd.concat(dfs)
data = data.pivot(columns='ticker', values=['open', 'high', 'low', 'close', 'volume'])

open_ = data['open']
high_ = data['high']
low_ = data['low']
close_ = data['close']
volume_ = data['volume']

def delay(df, n):
    return df.shift(n)

def rank(df):
    return df.rank(axis=1, pct=True)

def ts_rank(df, window):
    return df.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)

def delta(df, n):
    return df.diff(n)

def adv(df, n):
    return df.rolling(window=n).mean()

def correlation(df1, df2, window):
    return df1.rolling(window).corr(df2)

# 计算每日收益率returns
returns = close_.pct_change()

df_corr = correlation(open_, volume_, 10)
alpha = -1 * rank(delta(returns, 3)) * df_corr

# 4. 计算未来1日收益
future_return = close_.shift(-1) / close_ - 1

# 5. 因子分组与分组收益
quantiles = alpha.apply(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1, axis=1)
quantile_returns = pd.DataFrame(index=alpha.index, columns=range(1, 11))

for q in range(1, 11):
    mask = quantiles == q
    quantile_returns[q] = (future_return * mask).sum(axis=1) / mask.sum(axis=1)

# 6. 画累计收益曲线
log_cum_returns = np.log1p(quantile_returns).cumsum()

# 画累计收益曲线
plt.figure(figsize=(12, 6))
for col in log_cum_returns.columns:
    plt.plot(log_cum_returns.index, log_cum_returns[col], label=f"Quantile {col}")
plt.title("Cumulative Return by Quantile")
plt.xlabel("Date")
plt.ylabel("Log Cumulative Returns")
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_return_by_quantile.png')
plt.close()

# 统计每个分组的因子分布
factor_stats = alpha.stack().to_frame('factor')
factor_stats['quantile'] = quantiles.stack()
table = factor_stats.groupby('quantile')['factor'].agg(['min', 'max', 'mean', 'std', 'count'])
print(table)

# 分组因子均值柱状图
plt.figure(figsize=(8, 4))
plt.bar(table.index.astype(str), table['mean'])
plt.xlabel('Quantile')
plt.ylabel('Mean Factor Value')
plt.title('Mean Factor Value by Quantile')
plt.savefig('mean_factor_by_quantile.png')
plt.close()

# 以1日、5日、10日为例
periods = [1, 2, 3, 5, 10]
top_returns = []
bottom_returns = []
spread_returns = []
mean_returns_dict = {}

for p in periods:
    future_return_p = close_.shift(-p) / close_ - 1
    mean_returns = []
    for q in range(1, 11):
        mask = quantiles == q
        group_ret = (future_return_p * mask).sum(axis=1) / mask.sum(axis=1)
        mean_returns.append(group_ret.mean())
    mean_returns_dict[p] = mean_returns  # 保存每个period的分组均值
    top_returns.append(mean_returns[-1] * 10000)    # 10分组，bps
    bottom_returns.append(mean_returns[0] * 10000)  # 1分组，bps
    spread_returns.append((mean_returns[-1] - mean_returns[0]) * 10000)

# 组装成DataFrame
summary = pd.DataFrame({
    'Top Quantile (bps)': top_returns,
    'Bottom Quantile (bps)': bottom_returns,
    'Spread (bps)': spread_returns
}, index=[f'{p}D' for p in periods])
print(summary)

# 分组收益柱状图
plt.figure(figsize=(10, 5))
bar_width = 0.2
x = np.arange(1, 11)
for i, p in enumerate(periods):
    plt.bar(x + i*bar_width, mean_returns_dict[p], width=bar_width, label=f'{p}D')
plt.xlabel('Quantile')
plt.ylabel('Mean Return')
plt.title('Mean Return by Quantile and Holding Period')
plt.legend()
plt.savefig('mean_return_by_quantile_and_period.png')
plt.close()

table.to_csv('factor_quantile_stats.csv')

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table_img = ax.table(
    cellText=summary.round(3).values,
    colLabels=summary.columns,
    rowLabels=summary.index,
    loc='center',
    cellLoc='center'
)
table_img.auto_set_font_size(False)
table_img.set_fontsize(12)
table_img.scale(1.2, 1.2)
plt.title('收益表现', fontsize=16, pad=20)
plt.savefig('summary_table.png', bbox_inches='tight')
plt.close()

# 计算因子加权组合的每日收益
# 用因子值做权重（横截面标准化，防止极端值影响）
factor_weight = alpha.div(alpha.abs().sum(axis=1), axis=0).fillna(0)
# 组合每日收益 = 各股票未来1日收益 * 权重
portfolio_return = (future_return * factor_weight).sum(axis=1)
# 累计收益（普通复利，不是对数）
cumulative_return = (1 + portfolio_return).cumprod()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(cumulative_return.index, cumulative_return, color='seagreen', alpha=0.7)
plt.title('Factor Weighted Portfolio Cumulative Return (1D Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.axhline(1, color='black', linestyle='-')
plt.grid(True)
plt.tight_layout()
plt.savefig('factor_weighted_cum_return.png')
plt.close() 