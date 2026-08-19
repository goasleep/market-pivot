---
id: philosophy-position-sizing-001
title: 仓位与回撤控制框架
type: philosophy
author: 系统整理
source: A-Share Agent 内置方法论摘要
source_url: ""
published_at: ""
market_scope: [A-share]
asset_types: [ETF, LOF, stock]
horizon: short_medium
tags: [仓位, 回撤, 止损, 风险管理]
status: active
authority: methodology
---

# 核心观点

短中期基金交易的首要目标是控制单次错误和连续错误造成的组合回撤。仓位不
应只由看多程度决定，还应受到波动率、流动性、持仓集中度和最大可接受亏损
的约束。

# 适用条件

- 用户已经明确单只标的和组合的最大风险
- 止损、减仓和暂停交易条件可以被确定性计算
- 组合中不同标的的相关性被纳入考虑

# 失效条件

- 只设止损比例而不考虑跳空和流动性
- 把回撤控制误解为保证不亏损
- 在连续亏损后无规则地提高仓位

# 可检验假设

- 降低单标的最大仓位是否能改善组合最大回撤
- 在连续亏损后暂停新增仓位，是否能降低策略的尾部损失

# 可转化规则

仓位和止损应进入系统的确定性风险工具，不能仅保存在自然语言方法论中。
