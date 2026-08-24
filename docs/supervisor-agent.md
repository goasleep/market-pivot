# Supervisor Agent 架构

系统只存在一个当前架构，不维护 Agent v1/v2/v3/v4 的运行时兼容分支。

```text
用户请求
  -> Task Contract
  -> Supervisor 决策
       -> 直接回答
       -> 原子数据/计算/产物/模拟工具
       -> ResearchPlan 子能力
       -> 综合分析子图
  -> 工具或子能力结果返回 Supervisor
  -> Completion Judge
       -> 未完成：带缺口和下一行动返回 Supervisor
       -> 需用户信息：waiting_user / needs_input
       -> 已完成：输出最终答案和业务 outcome
```

改造包含以下 12 项：

1. 单一入口：所有对话请求进入 Supervisor，不再按基金题型选择根执行器。
2. 任务合同：入口生成 objective、deliverables、evidence requirements、工具需求和代表产品解析策略。
3. 模型选行动：Supervisor 大模型在统一工具面中自行选择直接完成、原子工具或子能力。
4. Supervisor 可亲自执行：简单问答、结果整合和最终建议不强制委派子 Agent。
5. 子能力工具化：ResearchPlan 和综合分析图均作为可选工具，执行结果必须返回 Supervisor。
6. 代表产品解析：宽泛基金类别默认主动寻找并披露可验证代表样本；找不到可靠候选才询问用户。
7. 数据缺口追踪：需要公开数据却没有成功工具证据时，Completion Judge 禁止结束。
8. 完成判定：模型无 tool call 只产生候选答案；Judge 决定继续、等待输入或终止。
9. 防止计划冒充结果：候选答案仍出现“下一步需要查询/进一步校准需”等未完成表述时自动继续。
10. 统一预算与失败语义：工具重试、超时、最大轮数和部分完成在同一循环中收敛。
11. 双轴状态：任务生命周期 status 与业务 outcome 分离，并通过存储、API、SSE 和前端展示。
12. 破坏性数据切换：`make reset-agent-data` 清除 Agent 对话、任务、事件、状态、产物和 checkpoint，保留设置、模拟盘、回测、策略与市场缓存。

数据重置默认仅预览。执行前停止 API/Agent worker，然后运行：

```bash
CONFIRM_AGENT_DATA_RESET=yes AGENT_WORKERS_STOPPED=yes \
  make reset-agent-data RESET_ARGS=--execute
```

验证以 `backend/tests/test_supervisor_end_to_end.py` 中的单一复杂基金比较问题为行为验收，不再使用 122 道基金题作为架构验收集。
