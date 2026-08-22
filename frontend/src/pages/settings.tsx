import { useEffect, useState, type ReactNode } from "react";
import { AlertCircle, CheckCircle2, Loader2, Settings } from "lucide-react";

import { getLLMConfig, type LLMConfig } from "@/api";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const environmentVariables = [
  ["LLM_PROVIDER", "Provider adapter"],
  ["LLM_MODEL", "Model ID"],
  ["LLM_TEMPERATURE", "Generation temperature"],
  ["LLM_CONTEXT_WINDOW", "Model context window"],
  ["LLM_MAX_OUTPUT_TOKENS", "Maximum output tokens"],
  ["OPENAI_API_KEY", "Shared API credential"],
  ["OPENAI_BASE_URL", "OpenAI-compatible endpoint"],
] as const;

export function SettingsPage() {
  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    void getLLMConfig()
      .then((data) => {
        if (mounted) setConfig(data);
      })
      .catch((reason: unknown) => {
        if (mounted)
          setError(
            reason instanceof Error ? reason.message : "Failed to load config",
          );
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted-foreground">
          LLM 配置由环境变量统一管理，此页面仅展示当前生效状态。
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Settings className="h-4 w-4" />
            Effective LLM Configuration
          </CardTitle>
          <CardDescription>
            修改环境变量后重启后端服务，新的配置才会生效。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <StatusRow label="Configuration Source">
            <Badge variant="secondary">Environment</Badge>
          </StatusRow>
          <StatusRow label="API Key">
            {config?.api_key_set ? (
              <Badge className="bg-green-500/15 text-green-600 hover:bg-green-500/15">
                <CheckCircle2 className="mr-1 h-3 w-3" />
                Configured
              </Badge>
            ) : (
              <Badge variant="destructive">Not Set</Badge>
            )}
          </StatusRow>
          <StatusRow label="Provider">
            <code className="text-xs">{config?.provider_type || "—"}</code>
          </StatusRow>
          <StatusRow label="Model">
            <code className="text-xs">{config?.model || "—"}</code>
          </StatusRow>
          <StatusRow label="Temperature">
            <code className="text-xs">{config?.temperature ?? "—"}</code>
          </StatusRow>
          <StatusRow label="Max Output Tokens">
            <code className="text-xs">
              {config?.max_tokens.toLocaleString("en-US") || "—"}
            </code>
          </StatusRow>
          <StatusRow label="Context Window">
            <code className="text-xs">
              {config?.context_window.toLocaleString("en-US") || "—"}
            </code>
          </StatusRow>
          <StatusRow label="Base URL">
            <code className="max-w-[70%] truncate text-xs">
              {config?.base_url || "—"}
            </code>
          </StatusRow>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Environment Variables</CardTitle>
          <CardDescription>
            这些变量是 LLM 配置的唯一来源，不会保存到应用数据库。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {environmentVariables.map(([name, description]) => (
            <div
              key={name}
              className="flex items-center justify-between gap-4 rounded-md border bg-muted/40 px-3 py-2"
            >
              <code className="text-xs font-medium">{name}</code>
              <span className="text-right text-xs text-muted-foreground">
                {description}
              </span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function StatusRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b py-2 last:border-b-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}
