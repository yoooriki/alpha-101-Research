# Alpha101 因子回测框架

## 项目简介
这是一个用于美股市场的量化因子回测框架，主要用于测试和分析Alpha因子的有效性。该框架使用Python实现，集成了数据获取、因子计算、回测分析和可视化等功能。

## 环境要求
- Python 3.x
- 依赖包：
  - akshare
  - pandas
  - numpy
  - matplotlib

## 数据说明
- 数据来源：使用akshare获取美股日线数据
- 默认股票池：AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA, NFLX, INTC, IBM
- 回测区间：2022年全年数据
- 数据字段：开盘价、最高价、最低价、收盘价、成交量

## 因子策略
当前实现的Alpha因子策略：
- 基于3日收益率变化的排序
- 与10日成交量的相关性
- 最终因子 = -1 * rank(delta(returns, 3)) * correlation(open, volume, 10)

## 回测框架功能
1. 因子计算工具函数：
   - delay(df, n)：时间序列延迟
   - rank(df)：横截面排序
   - ts_rank(df, window)：时序排序
   - delta(df, n)：差分
   - adv(df, n)：移动平均
   - correlation(df1, df2, window)：相关性计算

2. 回测分析：
   - 因子分组（10分位）
   - 分组收益率计算
   - 累积收益曲线
   - 因子特征统计

## 输出结果
程序会生成以下分析图表：
1. cumulative_return_by_quantile.png：各分位数组合的累积收益曲线
2. mean_factor_by_quantile.png：各分位数的平均因子值分布
3. factor_quantile_stats.csv：因子分位数统计信息
4. 其他可视化结果

## 使用方法
1. 运行alpha_factor_backtest.py文件
2. 可以通过修改股票池、回测区间、因子计算逻辑来测试不同策略

## 注意事项
- 股票池可以根据需要进行替换
- 回测区间可以在代码中调整
- 因子计算逻辑可以根据研究需要进行修改

## 后续优化方向
1. 增加更多Alpha因子策略
2. 添加风险调整后的收益分析
3. 实现更多回测评估指标
4. 优化数据处理效率