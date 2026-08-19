---
id: paper-time-series-momentum-001
title: 时间序列动量的可检验研究摘要
type: paper
author: 文献摘要模板
source: 待补充原始论文与正式引用
source_url: ""
published_at: ""
market_scope: [global]
asset_types: [ETF, LOF, stock]
horizon: short_medium
tags: [论文, 时间序列动量, 动量, 回测]
status: active
authority: paper_summary
---

# 研究问题

过去一段时间的自身收益方向，是否能够作为未来一段时间收益方向的研究信号。

# 方法摘要

将标的过去一段窗口的累计收益转化为方向信号，再按预先定义的持有周期、交易
成本和仓位规则进行检验。该摘要不是原始论文全文，也不代表在 A 股 ETF 上已经
得到验证。

# 适用边界

- 必须重新选择适合 A 股 ETF 的样本区间和交易频率
- 必须处理停牌、流动性、费用、滑点和 T+1 约束
- 必须进行滚动或样本外验证

# 可检验假设

- 不同回看窗口和持有窗口对 ETF 的结果是否稳定
- 动量信号在趋势、震荡和快速反转市场中的差异
- 纳入交易成本后是否仍有足够的风险调整收益

# 局限性

论文中的市场、数据、执行条件和研究周期可能与 A 股基金交易不同。该材料
只能帮助 Agent 设计实验，不能直接形成交易信号。
