import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { getLLMConfig, updateLLMConfig, type LLMConfig } from "@/api";
import {
  Settings,
  Save,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

export function SettingsPage() {
  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Form fields
  const [profileId, setProfileId] = useState("");
  const [profileName, setProfileName] = useState("");
  const [providerType, setProviderType] = useState("openai_compatible");
  const [model, setModel] = useState("");
  const [temperature, setTemperature] = useState(0.3);
  const [maxTokens, setMaxTokens] = useState(8192);
  const [routingEnabled, setRoutingEnabled] = useState(false);
  const [chatRouteProfile, setChatRouteProfile] = useState("");
  const [chatRouteModel, setChatRouteModel] = useState("");
  const [analysisRouteProfile, setAnalysisRouteProfile] = useState("");
  const [analysisRouteModel, setAnalysisRouteModel] = useState("");

  const fetchConfig = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getLLMConfig();
      setConfig(data);
      const active = data.profiles[data.active_profile_id];
      setProfileId(data.active_profile_id);
      setProfileName(active?.name || data.active_profile_id);
      setProviderType(active?.type || "openai_compatible");
      setModel(active?.model || "");
      setTemperature(active?.temperature ?? 0.3);
      setMaxTokens(active?.max_tokens ?? 8192);
      setRoutingEnabled(data.routing?.enabled ?? false);
      setChatRouteProfile(
        data.routing?.routes?.chat?.profile_id || data.active_profile_id,
      );
      setChatRouteModel(
        data.routing?.routes?.chat?.model || active?.model || "",
      );
      setAnalysisRouteProfile(
        data.routing?.routes?.analysis?.profile_id || data.active_profile_id,
      );
      setAnalysisRouteModel(
        data.routing?.routes?.analysis?.model || active?.model || "",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load config");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  const selectProfile = (id: string) => {
    const selected = config?.profiles[id];
    if (!selected) return;
    setProfileId(id);
    setProfileName(selected.name);
    setProviderType(selected.type);
    setModel(selected.model);
    setTemperature(selected.temperature);
    setMaxTokens(selected.max_tokens);
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(false);
    try {
      const update: Record<string, unknown> = {
        active_profile_id: profileId,
        profile_id: profileId,
        profile_name: profileName,
        provider_type: providerType,
        model,
        temperature,
        max_tokens: maxTokens,
        routing: {
          enabled: routingEnabled,
          routes: {
            chat: { profile_id: chatRouteProfile, model: chatRouteModel },
            analysis: {
              profile_id: analysisRouteProfile,
              model: analysisRouteModel,
            },
          },
        },
      };
      const data = await updateLLMConfig(update);
      setConfig(data);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save config");
    } finally {
      setSaving(false);
    }
  };

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
          Configure LLM provider and model parameters
        </p>
      </div>

      {/* Status banner */}
      {success && (
        <div className="flex items-center gap-2 rounded-md border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-600">
          <CheckCircle2 className="h-4 w-4" />
          Configuration saved successfully. Changes take effect immediately.
        </div>
      )}
      {error && (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {/* Connection status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Settings className="h-4 w-4" />
            LLM Provider Status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">API Key</span>
            {config?.api_key_set ? (
              <Badge className="bg-green-500/15 text-green-600 hover:bg-green-500/15">
                Configured
              </Badge>
            ) : (
              <Badge variant="destructive">Not Set</Badge>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">
              API Key Variable
            </span>
            <code className="text-xs text-muted-foreground">
              OPENAI_API_KEY
            </code>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm text-muted-foreground">Base URL</span>
            <code className="truncate text-xs text-muted-foreground">
              {config?.base_url}
            </code>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">
              Base URL Variable
            </span>
            <code className="text-xs text-muted-foreground">
              OPENAI_BASE_URL
            </code>
          </div>
        </CardContent>
      </Card>

      {/* Configuration form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Configuration</CardTitle>
          <CardDescription>
            Model settings are saved to the application database. Connection
            settings come only from environment variables.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="profile">Provider Profile</Label>
            <select
              id="profile"
              value={profileId}
              onChange={(e) => selectProfile(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
            >
              {config &&
                Object.values(config.profiles).map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}（{profile.type}）
                  </option>
                ))}
            </select>
            <p className="text-xs text-muted-foreground">
              所有 Profile 共用 OPENAI_API_KEY 和 OPENAI_BASE_URL，并分别配置
              Provider Type 和所选模型。
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="profile_name">Profile Name</Label>
            <Input
              id="profile_name"
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="provider_type">Provider Type</Label>
            <select
              id="provider_type"
              value={providerType}
              onChange={(e) => setProviderType(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
            >
              <option value="deepseek">DeepSeek Native</option>
              <option value="openai_compatible">OpenAI Compatible</option>
            </select>
          </div>

          <Separator />

          {/* Shared connection environment variables */}
          <div className="space-y-2">
            <Label>Connection Environment Variables</Label>
            <div className="space-y-1 rounded-md border bg-muted px-3 py-2 text-sm">
              <code className="block">OPENAI_API_KEY</code>
              <code className="block">OPENAI_BASE_URL</code>
            </div>
            <p className="text-xs text-muted-foreground">
              All providers use these environment variables. Set them before
              starting the backend; neither value is persisted here.
            </p>
          </div>

          <Separator />

          {/* Read-only Base URL */}
          <div className="space-y-2">
            <Label>Effective Base URL</Label>
            <code className="block overflow-x-auto rounded-md border bg-muted px-3 py-2 text-sm">
              {config?.base_url}
            </code>
            <p className="text-xs text-muted-foreground">
              Read from OPENAI_BASE_URL. Update the environment variable and
              restart the backend to use a proxy or self-hosted endpoint.
            </p>
          </div>

          {/* Model */}
          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Input
              id="model"
              list={`models-${profileId}`}
              value={model}
              onChange={(e) => {
                const val = e.target.value;
                setModel(val);
                // Auto-fill model defaults when switching
                const models = config?.profiles[profileId]?.available_models;
                if (models?.[val]) {
                  setMaxTokens(models[val].max_tokens);
                  if (models[val].temperature !== undefined)
                    setTemperature(models[val].temperature);
                }
              }}
              placeholder="输入模型名称，也可从预置列表选择"
            />
            <datalist id={`models-${profileId}`}>
              {config?.profiles[profileId]?.available_models &&
                Object.entries(config.profiles[profileId].available_models).map(
                  ([key, info]) => (
                    <option key={key} value={key} label={info.description} />
                  ),
                )}
            </datalist>
            <p className="text-xs text-muted-foreground">
              OpenAI-compatible 服务可直接填写服务端实际暴露的模型 ID。
            </p>
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <Label htmlFor="temperature">
              Temperature:{" "}
              <span className="text-muted-foreground">{temperature}</span>
            </Label>
            <input
              id="temperature"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0 (deterministic)</span>
              <span>2 (creative)</span>
            </div>
          </div>

          {/* Max tokens */}
          <div className="space-y-2">
            <Label htmlFor="max_tokens">Max Output Tokens</Label>
            <Input
              id="max_tokens"
              type="number"
              min="256"
              max="128000"
              step="256"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value) || 8192)}
            />
            <p className="text-xs text-muted-foreground">
              Maximum tokens the model can generate per response.
            </p>
          </div>

          <Separator />

          <div className="space-y-3 rounded-md border p-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="routing_enabled">Automatic Model Routing</Label>
                <p className="text-xs text-muted-foreground">
                  按聊天类型选择默认模型；聊天窗口仍可单次覆盖。
                </p>
              </div>
              <input
                id="routing_enabled"
                type="checkbox"
                checked={routingEnabled}
                onChange={(e) => setRoutingEnabled(e.target.checked)}
                className="h-4 w-4 accent-primary"
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label htmlFor="chat_route">General Chat Model</Label>
                <select
                  id="chat_route"
                  value={`${chatRouteProfile}:${chatRouteModel}`}
                  onChange={(e) => {
                    const [profile, ...rest] = e.target.value.split(":");
                    setChatRouteProfile(profile);
                    setChatRouteModel(rest.join(":"));
                  }}
                  disabled={!routingEnabled}
                  className="h-9 w-full rounded-md border border-input bg-transparent px-2 text-xs"
                >
                  {config &&
                    Object.values(config.profiles).flatMap((profile) =>
                      Object.keys(profile.available_models).map((modelName) => (
                        <option
                          key={`chat-${profile.id}-${modelName}`}
                          value={`${profile.id}:${modelName}`}
                        >
                          {profile.name} / {modelName}
                        </option>
                      )),
                    )}
                </select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="analysis_route">Analysis Model</Label>
                <select
                  id="analysis_route"
                  value={`${analysisRouteProfile}:${analysisRouteModel}`}
                  onChange={(e) => {
                    const [profile, ...rest] = e.target.value.split(":");
                    setAnalysisRouteProfile(profile);
                    setAnalysisRouteModel(rest.join(":"));
                  }}
                  disabled={!routingEnabled}
                  className="h-9 w-full rounded-md border border-input bg-transparent px-2 text-xs"
                >
                  {config &&
                    Object.values(config.profiles).flatMap((profile) =>
                      Object.keys(profile.available_models).map((modelName) => (
                        <option
                          key={`analysis-${profile.id}-${modelName}`}
                          value={`${profile.id}:${modelName}`}
                        >
                          {profile.name} / {modelName}
                        </option>
                      )),
                    )}
                </select>
              </div>
            </div>
          </div>

          <Separator />

          {/* Save button */}
          <div className="flex justify-end gap-3">
            <Button variant="outline" onClick={fetchConfig} disabled={saving}>
              Reset
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save Configuration
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
