import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { streamAnalysis } from "@/api";
import type { AnalysisResult, SSEProgress } from "@/types";
import { Loader2, Search } from "lucide-react";

const STAGE_LABELS: Record<string, string> = {
  market_data: "市场数据获取",
  technical: "技术分析",
  fundamentals: "基本面分析",
  sentiment: "舆情分析",
  debate: "多空辩论",
  risk: "风控评估",
  portfolio: "投资组合决策",
};

export function AnalysisPage() {
  const [ticker, setTicker] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SSEProgress[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = () => {
    if (!ticker.trim()) return;
    setRunning(true);
    setProgress([]);
    setResult(null);
    setError(null);

    streamAnalysis(
      ticker.trim(),
      (data) => setProgress((prev) => [...prev, data]),
      (data) => {
        setResult(data);
        setRunning(false);
      },
      () => {
        setError("连接中断");
        setRunning(false);
      }
    );
  };

  const progressPct = running ? (progress.length / 7) * 100 : result ? 100 : 0;

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Agent Analysis</h1>
        <p className="text-sm text-muted-foreground">输入股票代码，启动多智能体协作分析</p>
      </div>

      {/* Input */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">股票分析</CardTitle>
          <CardDescription>输入 A 股股票代码（如 000737、600519）</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-end gap-3">
            <div className="flex-1 space-y-2">
              <Label htmlFor="ticker">股票代码</Label>
              <Input
                id="ticker"
                placeholder="000737"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !running && handleRun()}
                disabled={running}
              />
            </div>
            <Button onClick={handleRun} disabled={running || !ticker.trim()}>
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              {running ? "分析中..." : "开始分析"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Progress */}
      {(running || progress.length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Agent 工作进度</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Progress value={progressPct} />
            <div className="space-y-2">
              {progress.map((p, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <Badge variant="success" className="h-5 w-5 justify-center p-0 text-xs">
                    {i + 1}
                  </Badge>
                  <span className="text-muted-foreground">{STAGE_LABELS[p.stage] || p.stage}</span>
                  <span>{p.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Result */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">分析结果 - {result.ticker}</CardTitle>
              <Badge
                variant={
                  result.decision === "buy" ? "success" :
                  result.decision === "sell" ? "destructive" : "secondary"
                }
                className="text-sm"
              >
                {result.decision === "buy" ? "买入" : result.decision === "sell" ? "卖出" : "持有"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <Label className="text-xs text-muted-foreground">置信度</Label>
                <p className="text-lg font-semibold">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">目标价</Label>
                <p className="text-lg font-semibold">
                  {result.target_price ? `¥${result.target_price}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">止损价</Label>
                <p className="text-lg font-semibold">
                  {result.stop_loss ? `¥${result.stop_loss}` : "—"}
                </p>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">建议仓位</Label>
                <p className="text-lg font-semibold">
                  {result.position_size ? `${(result.position_size * 100).toFixed(0)}%` : "—"}
                </p>
              </div>
            </div>
            <Separator />
            <div>
              <Label className="text-xs text-muted-foreground">决策推理</Label>
              <p className="mt-1 text-sm leading-relaxed">{result.reasoning}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
