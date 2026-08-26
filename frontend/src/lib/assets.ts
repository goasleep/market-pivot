import type { AssetType } from "@/types";

export function assetTypeLabel(value?: AssetType | string | null): string {
  if (value === "stock") return "股票";
  if (value === "etf") return "ETF";
  if (value === "lof") return "LOF";
  if (value === "open_fund") return "场外基金";
  return "未知产品";
}
