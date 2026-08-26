先确定单一产品类别，再发现并核验候选。货币基金走 money_yield，其他类别走 NAV；不得跨类别直接排名。完成候选指标后必须调用 `screen_compare_open_funds`，只有工具返回 `ranking_is_formal=true` 才能给出正式排名；缺少身份核验、数据日期、类别一致性或正式评分证据时，只返回条件性候选和结构化缺口。
