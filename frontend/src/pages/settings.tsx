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
import {
  getLLMConfig,
  updateLLMConfig,
  type LLMConfig,
} from "@/api";
import { Settings, Save, Loader2, CheckCircle2, AlertCircle, Eye, EyeOff } from "lucide-react";

export function SettingsPage() {
  const [config, setConfig] = useState<LLMConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Form fields
  const [apiKey, setApiKey] = useState("");
  const [profileId, setProfileId] = useState("");
  const [profileName, setProfileName] = useState("");
  const [providerType, setProviderType] = useState("openai_compatible");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [temperature, setTemperature] = useState(0.3);
  const [maxTokens, setMaxTokens] = useState(8192);
  const [showApiKey, setShowApiKey] = useState(false);
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
      setBaseUrl(active?.base_url || data.base_url);
      setModel(active?.model || data.model);
      setTemperature(active?.temperature ?? data.temperature);
      setMaxTokens(active?.max_tokens ?? data.max_tokens);
      setRoutingEnabled(data.routing?.enabled ?? false);
      setChatRouteProfile(data.routing?.routes?.chat?.profile_id || data.active_profile_id);
      setChatRouteModel(data.routing?.routes?.chat?.model || active?.model || data.model);
      setAnalysisRouteProfile(data.routing?.routes?.analysis?.profile_id || data.active_profile_id);
      setAnalysisRouteModel(data.routing?.routes?.analysis?.model || active?.model || data.model);
      setApiKey(""); // never pre-fill with actual key
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
    setBaseUrl(selected.base_url);
    setModel(selected.model);
    setTemperature(selected.temperature);
    setMaxTokens(selected.max_tokens);
    setApiKey("");
  };

  const selectedProfile = config?.profiles[profileId];

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
        base_url: baseUrl,
        model,
        temperature,
        max_tokens: maxTokens,
        routing: {
          enabled: routingEnabled,
          routes: {
            chat: { profile_id: chatRouteProfile, model: chatRouteModel },
            analysis: { profile_id: analysisRouteProfile, model: analysisRouteModel },
          },
        },
      };
      // Only send api_key if user typed a new one
      if (apiKey.trim()) {
        update.api_key = apiKey.trim();
      }

      const data = await updateLLMConfig(update);
      setConfig(data);
      setApiKey("");
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
            {selectedProfile?.api_key_set ? (
              <Badge className="bg-green-500/15 text-green-600 hover:bg-green-500/15">
                Configured
              </Badge>
            ) : (
              <Badge variant="destructive">Not Set</Badge>
            )}
          </div>
          {selectedProfile?.api_key_set && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Masked Key</span>
              <code className="text-xs text-muted-foreground">
                {selectedProfile.api_key_masked}
              </code>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Configuration form */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Configuration</CardTitle>
          <CardDescription>
            Settings are saved to a local config file and hot-reloaded without
            server restart.
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
              {config && Object.values(config.profiles).map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}（{profile.type}）
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              每个 Profile 可以拥有独立的 API Key、Base URL 和模型目录。
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="profile_name">Profile Name</Label>
            <Input id="profile_name" value={profileName} onChange={(e) => setProfileName(e.target.value)} />
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

          {/* API Key */}
          <div className="space-y-2">
            <Label htmlFor="api_key">API Key</Label>
            <div className="flex gap-2">
              <Input
                id="api_key"
                type={showApiKey ? "text" : "password"}
                placeholder={
                  selectedProfile?.api_key_set
                    ? `Current: ${selectedProfile.api_key_masked} (type new key to replace)`
                    : "Enter your provider API key"
                }
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="flex-1"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowApiKey(!showApiKey)}
                type="button"
              >
                {showApiKey ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Leave empty to keep the existing key. Get your key from{" "}
              <a
                href={providerType === "deepseek" ? "https://platform.deepseek.com/api_keys" : "https://platform.openai.com/api-keys"}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                provider platform
              </a>
              .
            </p>
          </div>

          <Separator />

          {/* Base URL */}
          <div className="space-y-2">
            <Label htmlFor="base_url">Base URL</Label>
            <Input
              id="base_url"
              type="text"
              placeholder="https://api.deepseek.com/v1"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              OpenAI-compatible endpoint. Change this to use a proxy or
              self-hosted endpoint.
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
                  if (models[val].temperature !== undefined) setTemperature(models[val].temperature);
                }
              }}
              placeholder="输入模型名称，也可从预置列表选择"
            />
            <datalist id={`models-${profileId}`}>
              {config?.profiles[profileId]?.available_models &&
                Object.entries(config.profiles[profileId].available_models).map(([key, info]) => (
                  <option key={key} value={key} label={info.description} />
                ))}
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
              max="65536"
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
                <p className="text-xs text-muted-foreground">按聊天类型选择默认模型；聊天窗口仍可单次覆盖。</p>
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
                  {config && Object.values(config.profiles).flatMap((profile) =>
                    Object.keys(profile.available_models).map((modelName) => (
                      <option key={`chat-${profile.id}-${modelName}`} value={`${profile.id}:${modelName}`}>
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
                  {config && Object.values(config.profiles).flatMap((profile) =>
                    Object.keys(profile.available_models).map((modelName) => (
                      <option key={`analysis-${profile.id}-${modelName}`} value={`${profile.id}:${modelName}`}>
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
