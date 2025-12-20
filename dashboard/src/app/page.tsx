/**
 * Copyright (C) 2024 Delia Contributors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

"use client"

import { useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { XAxis, YAxis, Cell, CartesianGrid, LineChart, Line, ResponsiveContainer, Tooltip, BarChart, Bar } from "recharts"

interface ModelStats {
  calls: number
  tokens: number
}

interface UsageStats {
  quick: ModelStats
  coder: ModelStats
  moe: ModelStats
  thinking: ModelStats
}

interface RecentCall {
  timestamp: string
  model: string
  task_type: string
  language: string
  tokens: number
  elapsed_ms: number
  preview: string
  thinking: boolean
  backend_type?: string
  backend?: string
}

interface ResponseTime {
  ts: string
  ms: number
}

interface EnhancedStats {
  task_stats: Record<string, number>
  recent_calls: RecentCall[]
  response_times: {
    quick: ResponseTime[]
    coder: ResponseTime[]
    moe: ResponseTime[]
    thinking: ResponseTime[]
  }
}

interface BackendModels {
  quick: string
  coder: string
  moe: string
  thinking: string
}

interface BackendHealth {
  available: boolean
  response_time_ms: number
  last_error: string
  loaded_models: string[]
  circuit_open: boolean
  consecutive_failures?: number
}

interface BackendStatus {
  id: string
  name: string
  provider: "ollama" | "llamacpp" | "lmstudio" | "vllm" | "openai" | "gemini" | "custom"
  type: "local" | "remote"
  url: string
  enabled: boolean
  priority: number
  models: BackendModels
  health: BackendHealth
  api_key?: string
}

interface BackendsResponse {
  backends: BackendStatus[]
  routing: {
    prefer_local: boolean
    fallback_enabled: boolean
    load_balance: boolean
  }
  activeBackend: string
  summary: {
    total: number
    enabled: number
    available: number
    local: number
    remote: number
  }
  timestamp: string
}

interface CircuitBreakerStatus {
  available: boolean
  consecutive_failures: number
  last_error: string
  circuit_open: boolean
  seconds_until_available: number
  safe_context_kb: number
}

interface CircuitBreakerState {
  ollama: CircuitBreakerStatus
  llamacpp: CircuitBreakerStatus
  active_backend: string
  timestamp: string
}

interface HedgingConfig {
  enabled: boolean
  delay_ms: number
  max_backends: number
}

interface PrewarmConfig {
  enabled: boolean
  threshold: number
  check_interval_minutes: number
}

interface AffinityPair {
  backend: string
  task: string
  score: number
}

interface MelonLeaderboardEntry {
  model_id: string
  task_type: string
  melons: number
  golden_melons: number
  total_melon_value: number
  success_rate: number
  total_responses: number
}

interface MelonsResponse {
  leaderboard: MelonLeaderboardEntry[]
  totals: {
    melons: number
    golden_melons: number
    total_value: number
    models: number
  }
  timestamp: string
}

interface RoutingIntelligence {
  hedging: HedgingConfig
  prewarm: PrewarmConfig
  affinity: {
    alpha: number
    trackedPairs: number
    topPairs: AffinityPair[]
  }
  prewarmStatus: {
    alpha: number
    threshold: number
    trackedEntries: number
    currentHour: number
    currentPredictions: string[]
    nextHourPredictions: string[]
  }
  timestamp: string
}

const PROVIDER_COLORS: Record<string, string> = {
  ollama: "#689B8A",
  llamacpp: "#4A7D6D",
  lmstudio: "#5A8FBA",
  vllm: "#8BB5A6",
  openai: "#1E3A32",
  gemini: "#FF6B7A",
  custom: "#6B8F85",
}

const TYPE_COLORS: Record<string, string> = {
  local: "#689B8A",
  remote: "#FF6B7A",
}

const COLORS = {
  quick: "#8BB5A6",
  coder: "#689B8A",
  moe: "#FF6B7A",
  thinking: "#E85566",
}

const TASK_COLORS: Record<string, string> = {
  review: "#8BB5A6",
  analyze: "#689B8A",
  generate: "#4A7D6D",
  summarize: "#A8D4C4",
  critique: "#FF6B7A",
  quick: "#8BB5A6",
  plan: "#FFB3BA",
  think: "#E85566",
  other: "#6B8F85",
}

function WatermelonSeed({ className = "", style = {} }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 10 16" className={className} style={style} fill="currentColor">
      <ellipse cx="5" cy="8" rx="4" ry="7" />
    </svg>
  )
}

const normalizeModel = (model: string | undefined | null): "quick" | "coder" | "moe" | "thinking" => {
  if (!model) return "quick"
  const m = model.toLowerCase()
  if (m === "14b" || m === "quick") return "quick"
  if (m === "30b" || m === "coder") return "coder"
  if (m === "moe") return "moe"
  if (m === "thinking" || m.includes("think") || m.includes("reason")) return "thinking"
  return "quick"
}

type TabType = "activity" | "backends" | "analytics" | "logs"

export default function Dashboard() {
  const [stats, setStats] = useState<UsageStats | null>(null)
  const [enhanced, setEnhanced] = useState<EnhancedStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">("dark")
  const [activeTab, setActiveTab] = useState<TabType>("activity")
  const [ollamaLogs, setOllamaLogs] = useState<string[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [backendsResponse, setBackendsResponse] = useState<BackendsResponse | null>(null)
  const [activeBackend, setActiveBackend] = useState<string>("")
  const [secondsAgo, setSecondsAgo] = useState(0)
  const [showAddBackend, setShowAddBackend] = useState(false)
  const [editingBackend, setEditingBackend] = useState<BackendStatus | null>(null)
  const [circuitBreaker, setCircuitBreaker] = useState<CircuitBreakerState | null>(null)
  const [selectedLogProvider, setSelectedLogProvider] = useState<string>("all")
  const [routingIntelligence, setRoutingIntelligence] = useState<RoutingIntelligence | null>(null)
  const [melonsData, setMelonsData] = useState<MelonsResponse | null>(null)

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000))
    }, 1000)
    return () => clearInterval(timer)
  }, [lastUpdated])

  useEffect(() => {
    const stored = localStorage.getItem("delia-theme") as "light" | "dark" | null
    if (stored) {
      setTheme(stored)
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setTheme("dark")
    }
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    localStorage.setItem("delia-theme", theme)
  }, [theme])

  const toggleTheme = () => setTheme(prev => prev === "dark" ? "light" : "dark")

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/stats")
      if (!res.ok) throw new Error("Failed to fetch stats")
      const data = await res.json()
      setStats(data.stats)
      setEnhanced(data.enhanced)
      setError(null)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const eventSource = new EventSource("/api/sse/backends")
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as BackendsResponse
        if (data.backends) {
          setBackendsResponse(data)
          setActiveBackend(data.activeBackend || "")
          setLastUpdated(new Date())
        }
      } catch { /* ignore */ }
    }
    eventSource.onerror = () => console.debug("SSE reconnecting...")
    return () => eventSource.close()
  }, [])

  const fetchCircuitBreaker = useCallback(async () => {
    try {
      const res = await fetch("/api/circuit-breaker")
      if (res.ok) {
        const data = await res.json()
        if (data.success) setCircuitBreaker(data)
      }
    } catch { /* ignore */ }
  }, [])

  const fetchRoutingIntelligence = useCallback(async () => {
    try {
      const res = await fetch("/api/routing")
      if (res.ok) setRoutingIntelligence(await res.json())
    } catch { /* ignore */ }
  }, [])

  const fetchMelons = useCallback(async () => {
    try {
      const res = await fetch("/api/melons?limit=5")
      if (res.ok) setMelonsData(await res.json())
    } catch { /* ignore */ }
  }, [])

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (selectedLogProvider === "all") params.set("all", "true")
      else params.set("backend_id", selectedLogProvider)
      const res = await fetch(`/api/logs?${params}`)
      if (res.ok) {
        const data = await res.json()
        if (data.logs?.length > 0) {
          const formattedLogs = data.logs.map((log: { type: string; message: string; garden_msg?: string; backend?: string; backend_id?: string; provider?: string; model?: string; tokens?: number }) => {
            const displayMsg = log.garden_msg || log.message
            const backend = log.backend || log.backend_id || log.provider || ""
            const tokenInfo = log.tokens ? ` (${log.tokens} tokens)` : ""
            const modelInfo = log.model ? ` [${log.model}]` : ""
            return backend ? `[${backend}]${modelInfo} ${displayMsg}${tokenInfo}` : `${displayMsg}${modelInfo}${tokenInfo}`
          })
          setOllamaLogs(formattedLogs)
          setIsStreaming(data.logs.some((log: { type: string }) => log.type === "STREAM"))
        } else {
          setOllamaLogs([])
          setIsStreaming(false)
        }
      }
    } catch { /* ignore */ }
  }, [selectedLogProvider])

  useEffect(() => {
    fetchStats()
    fetchCircuitBreaker()
    fetchRoutingIntelligence()
    fetchMelons()
    const interval = setInterval(() => {
      fetchStats()
      fetchCircuitBreaker()
      fetchRoutingIntelligence()
      fetchMelons()
    }, 10000)
    return () => clearInterval(interval)
  }, [fetchStats, fetchCircuitBreaker, fetchRoutingIntelligence, fetchMelons])

  useEffect(() => {
    if (activeTab === "logs") {
      fetchLogs()
      const interval = setInterval(fetchLogs, 1000)
      return () => clearInterval(interval)
    }
  }, [activeTab, fetchLogs])

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 rounded-full border-4 border-primary border-t-transparent animate-spin" />
          <span className="text-muted-foreground text-sm">Loading...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="max-w-sm">
          <CardHeader>
            <CardTitle className="text-destructive">Connection Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <button onClick={fetchStats} className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm">
              Retry
            </button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const totalCalls = (stats?.quick?.calls || 0) + (stats?.coder?.calls || 0) + (stats?.moe?.calls || 0) + (stats?.thinking?.calls || 0)
  const totalTokens = (stats?.quick?.tokens || 0) + (stats?.coder?.tokens || 0) + (stats?.moe?.tokens || 0) + (stats?.thinking?.tokens || 0)
  const backendStats = (enhanced?.recent_calls || []).reduce((acc, call) => {
    const type = (call.backend_type || "local") as "local" | "remote"
    acc[type].calls += 1
    acc[type].tokens += call.tokens || 0
    return acc
  }, { local: { calls: 0, tokens: 0 }, remote: { calls: 0, tokens: 0 } })

  const hasOpenCircuit = circuitBreaker && Object.entries(circuitBreaker).some(([key, status]) => 
    key !== 'active_backend' && key !== 'timestamp' && status?.circuit_open
  )

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Compact Header */}
      <header className="sticky top-0 z-50 border-b bg-card/95 backdrop-blur">
        <div className="container mx-auto px-4 h-12 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <img src="/logo.svg" alt="Delia" className="w-6 h-6" />
              <span className="font-semibold hidden sm:inline">Delia</span>
            </div>
            
            {/* Quick Stats */}
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1">
                <span className="font-semibold">{totalCalls.toLocaleString()}</span>
                <span className="text-muted-foreground">calls</span>
              </span>
              <span className="text-muted-foreground">·</span>
              <span className="flex items-center gap-1">
                <span className="font-semibold">{totalTokens >= 1000000 ? `${(totalTokens/1000000).toFixed(1)}M` : totalTokens >= 1000 ? `${(totalTokens/1000).toFixed(0)}k` : totalTokens}</span>
                <span className="text-muted-foreground">tokens</span>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Backend Status Dots */}
            <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-muted/30">
              {backendsResponse?.backends?.slice(0, 5).map((b) => (
                <div
                  key={b.id}
                  className={`w-2 h-2 rounded-full ${b.health.available ? activeBackend === b.id ? 'ring-2 ring-offset-1 ring-offset-background ring-primary' : '' : 'opacity-30'}`}
                  style={{ backgroundColor: PROVIDER_COLORS[b.provider] || PROVIDER_COLORS.custom }}
                  title={`${b.name}: ${b.health.available ? 'online' : 'offline'}`}
                />
              ))}
              {hasOpenCircuit && <span className="text-destructive ml-1" title="Circuit breaker open">⚠</span>}
            </div>
            
            <span className="text-xs text-muted-foreground hidden sm:inline">{secondsAgo}s</span>
            
            <button onClick={toggleTheme} className="p-1.5 rounded hover:bg-accent transition-colors" title="Toggle theme">
              {theme === "dark" ? (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="border-b bg-card/50">
        <div className="container mx-auto px-4">
          <div className="flex gap-1">
            {([
              { id: "activity", label: "Activity", icon: "📊" },
              { id: "backends", label: "Backends", icon: "🌱" },
              { id: "analytics", label: "Analytics", icon: "📈" },
              { id: "logs", label: "Logs", icon: "📜" },
            ] as const).map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  activeTab === tab.id
                    ? "border-primary text-primary"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <span className="hidden sm:inline mr-1">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-4">
        {activeTab === "activity" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Recent Activity - Takes 2 columns */}
            <Card className="lg:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <WatermelonSeed className="w-3 h-4 text-primary" />
                  Recent Harvests
                </CardTitle>
              </CardHeader>
              <CardContent>
                {enhanced?.recent_calls?.length ? (
                  <div className="space-y-1 max-h-[400px] overflow-y-auto">
                    {[...enhanced.recent_calls].reverse().slice(0, 20).map((call, idx) => {
                      const tier = normalizeModel(call.model)
                      const isRemote = call.backend_type === "remote"
                      return (
                        <div key={idx} className="flex items-center gap-2 py-1.5 px-2 rounded text-sm hover:bg-muted/50">
                          <WatermelonSeed className="w-2 h-3 flex-shrink-0" style={{ color: COLORS[tier] }} />
                          <Badge variant="outline" className="text-[10px] px-1 py-0" style={{ borderColor: COLORS[tier], color: COLORS[tier] }}>
                            {tier}
                          </Badge>
                          <span className="text-xs px-1 rounded" style={{ color: isRemote ? "#FF6B7A" : "#689B8A" }}>
                            {isRemote ? "☁️" : "🌱"}
                          </span>
                          <span className="flex-1 truncate text-muted-foreground">{call.preview}</span>
                          <span className="text-xs text-muted-foreground whitespace-nowrap">
                            {call.tokens.toLocaleString()}t · {(call.elapsed_ms/1000).toFixed(1)}s
                          </span>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className="h-32 flex items-center justify-center text-muted-foreground text-sm">
                    No activity yet
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Right Column - Melons + Quick Stats */}
            <div className="space-y-4">
              {/* Melon Leaderboard */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    🍈 Leaderboard
                    {melonsData?.totals?.golden_melons ? (
                      <span className="text-xs font-normal text-muted-foreground">
                        {melonsData.totals.golden_melons}🏆
                      </span>
                    ) : null}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {melonsData?.leaderboard?.length ? (
                    <div className="space-y-1.5">
                      {melonsData.leaderboard.slice(0, 5).map((entry, idx) => (
                        <div key={`${entry.model_id}-${entry.task_type}`} className="flex items-center gap-2 text-sm">
                          <span className="w-5 text-center">{idx < 3 ? ["🥇", "🥈", "🥉"][idx] : `${idx + 1}.`}</span>
                          <span className="flex-1 truncate font-medium">{entry.model_id.split(':')[0]}</span>
                          <span className="text-xs text-muted-foreground">
                            {entry.golden_melons > 0 && `${entry.golden_melons}🏆 `}
                            {entry.melons}🍈
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="h-20 flex items-center justify-center text-muted-foreground text-sm">
                      No melons earned yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Model Tier Usage */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Model Usage</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {(["quick", "coder", "moe", "thinking"] as const).map((tier) => {
                      const tierStats = stats?.[tier]
                      const pct = totalCalls > 0 ? ((tierStats?.calls || 0) / totalCalls * 100) : 0
                      return (
                        <div key={tier} className="flex items-center gap-2">
                          <span className="w-16 text-xs capitalize" style={{ color: COLORS[tier] }}>{tier}</span>
                          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: COLORS[tier] }} />
                          </div>
                          <span className="w-12 text-xs text-right text-muted-foreground">{tierStats?.calls || 0}</span>
                        </div>
                      )
                    })}
                  </div>
                  <div className="flex justify-between mt-3 pt-2 border-t text-xs text-muted-foreground">
                    <span>🌱 {backendStats.local.calls} local</span>
                    <span>☁️ {backendStats.remote.calls} remote</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === "backends" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Backends List */}
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <WatermelonSeed className="w-3 h-4" />
                    Gardens
                  </CardTitle>
                  <button
                    onClick={() => setShowAddBackend(true)}
                    className="text-xs px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 text-primary"
                  >
                    + Add
                  </button>
                </div>
                <CardDescription className="text-xs">
                  {backendsResponse?.summary
                    ? `${backendsResponse.summary.available}/${backendsResponse.summary.enabled} available`
                    : "Loading..."}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {backendsResponse?.backends?.map((backend) => {
                  const isActive = activeBackend === backend.id
                  const isHealthy = backend.health.available
                  return (
                    <div
                      key={backend.id}
                      onClick={() => setEditingBackend(backend)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        isHealthy
                          ? isActive ? "border-primary/50 bg-primary/5" : "border-border hover:border-primary/30"
                          : "border-border/50 opacity-60"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${isHealthy && isActive ? 'animate-pulse' : ''}`}
                          style={{ backgroundColor: isHealthy ? (PROVIDER_COLORS[backend.provider] || '#689B8A') : '#666' }}
                        />
                        <span className="font-medium text-sm flex-1">{backend.name}</span>
                        <Badge variant="outline" className="text-[10px]" style={{ borderColor: TYPE_COLORS[backend.type], color: TYPE_COLORS[backend.type] }}>
                          {backend.type}
                        </Badge>
                        {backend.health.circuit_open && <span className="text-destructive" title="Circuit open">⚠</span>}
                      </div>
                      <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                        <span>{backend.provider}</span>
                        <span>·</span>
                        <span className="truncate">{backend.url}</span>
                        {backend.health.response_time_ms > 0 && (
                          <>
                            <span>·</span>
                            <span>{backend.health.response_time_ms}ms</span>
                          </>
                        )}
                      </div>
                      {backend.health.loaded_models?.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {backend.health.loaded_models.slice(0, 3).map((m, i) => (
                            <Badge key={i} variant="secondary" className="text-[10px] px-1 py-0">{m}</Badge>
                          ))}
                          {backend.health.loaded_models.length > 3 && (
                            <span className="text-[10px] text-muted-foreground">+{backend.health.loaded_models.length - 3}</span>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
                {!backendsResponse?.backends?.length && (
                  <div className="h-32 flex items-center justify-center text-muted-foreground text-sm">
                    No backends configured
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Routing Config */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Routing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Mode</label>
                  <Select
                    value={backendsResponse?.routing?.load_balance ? "load_balance" : backendsResponse?.routing?.prefer_local ? "prefer_local" : "prefer_remote"}
                    onValueChange={async (value) => {
                      const updates = { prefer_local: value === "prefer_local", load_balance: value === "load_balance" }
                      try {
                        const res = await fetch("/api/backends", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(updates) })
                        if (res.ok) setBackendsResponse(prev => prev ? { ...prev, routing: { ...prev.routing, ...updates } } : null)
                      } catch { /* ignore */ }
                    }}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="prefer_local">Prefer Local</SelectItem>
                      <SelectItem value="prefer_remote">Prefer Remote</SelectItem>
                      <SelectItem value="load_balance">Load Balance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-3">
                  {[
                    { label: "Auto-Fallback", desc: "Try next backend if primary fails", value: backendsResponse?.routing?.fallback_enabled, key: "fallback" },
                    { label: "Hedged Requests", desc: "Race multiple backends", value: routingIntelligence?.hedging?.enabled, key: "hedging" },
                    { label: "Pre-warming", desc: "Pre-load predicted models", value: routingIntelligence?.prewarm?.enabled, key: "prewarm" },
                  ].map((item) => (
                    <div key={item.key} className="flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium">{item.label}</span>
                        <p className="text-xs text-muted-foreground">{item.desc}</p>
                      </div>
                      <button
                        onClick={async () => {
                          const newValue = !item.value
                          const endpoint = item.key === "fallback" ? "/api/backends" : "/api/routing"
                          const body = item.key === "fallback" 
                            ? { fallback_enabled: newValue }
                            : { [item.key]: { enabled: newValue } }
                          try {
                            const res = await fetch(endpoint, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
                            if (res.ok) {
                              if (item.key === "fallback") {
                                setBackendsResponse(prev => prev ? { ...prev, routing: { ...prev.routing, fallback_enabled: newValue } } : null)
                              } else {
                                fetchRoutingIntelligence()
                              }
                            }
                          } catch { /* ignore */ }
                        }}
                        className={`w-9 h-5 rounded-full transition-colors relative ${item.value ? "bg-primary" : "bg-muted"}`}
                      >
                        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${item.value ? "left-4" : "left-0.5"}`} />
                      </button>
                    </div>
                  ))}
                </div>

                {/* Circuit Breaker Alert */}
                {hasOpenCircuit && (
                  <div className="p-3 rounded-lg border border-destructive/50 bg-destructive/10">
                    <div className="flex items-center gap-2 text-sm font-medium text-destructive">
                      ⚠ Circuit Breaker Open
                    </div>
                    {Object.entries(circuitBreaker!).filter(([k, v]) => k !== 'active_backend' && k !== 'timestamp' && v?.circuit_open).map(([backend, status]) => (
                      <div key={backend} className="text-xs text-muted-foreground mt-1">
                        {backend}: {(status as CircuitBreakerStatus).consecutive_failures} failures
                        {(status as CircuitBreakerStatus).seconds_until_available > 0 && ` · ${Math.ceil((status as CircuitBreakerStatus).seconds_until_available)}s until retry`}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "analytics" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Task Distribution */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Task Types</CardTitle>
              </CardHeader>
              <CardContent>
                {enhanced?.task_stats && Object.values(enhanced.task_stats).some(v => v > 0) ? (
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={Object.entries(enhanced.task_stats).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({ name, value }))}
                        layout="vertical"
                        margin={{ top: 5, right: 30, bottom: 5, left: 60 }}
                      >
                        <XAxis type="number" tick={{ fontSize: 10 }} />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={55} />
                        <Tooltip formatter={(value: number) => [value, "calls"]} />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {Object.entries(enhanced.task_stats).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([name], i) => (
                            <Cell key={i} fill={TASK_COLORS[name] || TASK_COLORS.other} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-[250px] flex items-center justify-center text-muted-foreground text-sm">No data yet</div>
                )}
              </CardContent>
            </Card>

            {/* Response Times */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Response Times</CardTitle>
              </CardHeader>
              <CardContent>
                {enhanced?.response_times && (enhanced.response_times.quick?.length > 0 || enhanced.response_times.coder?.length > 0 || enhanced.response_times.moe?.length > 0) ? (
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={(() => {
                        const maxLen = Math.max(
                          enhanced.response_times.quick?.length || 0,
                          enhanced.response_times.coder?.length || 0,
                          enhanced.response_times.moe?.length || 0,
                          enhanced.response_times.thinking?.length || 0
                        )
                        return Array.from({ length: Math.min(maxLen, 20) }, (_, i) => ({
                          idx: i,
                          quick: enhanced.response_times.quick?.slice(-20)[i]?.ms,
                          coder: enhanced.response_times.coder?.slice(-20)[i]?.ms,
                          moe: enhanced.response_times.moe?.slice(-20)[i]?.ms,
                          thinking: enhanced.response_times.thinking?.slice(-20)[i]?.ms,
                        }))
                      })()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="idx" tick={false} />
                        <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}s`} width={35} />
                        <Tooltip formatter={(value: number) => [`${value.toLocaleString()}ms`, ""]} />
                        <Line type="monotone" dataKey="quick" stroke={COLORS.quick} strokeWidth={2} dot={false} connectNulls />
                        <Line type="monotone" dataKey="coder" stroke={COLORS.coder} strokeWidth={2} dot={false} connectNulls />
                        <Line type="monotone" dataKey="moe" stroke={COLORS.moe} strokeWidth={2} dot={false} connectNulls />
                        <Line type="monotone" dataKey="thinking" stroke={COLORS.thinking} strokeWidth={2} dot={false} connectNulls />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-[250px] flex items-center justify-center text-muted-foreground text-sm">No data yet</div>
                )}
              </CardContent>
            </Card>

            {/* Learning Status */}
            {routingIntelligence && (routingIntelligence.affinity?.trackedPairs > 0 || routingIntelligence.prewarmStatus?.trackedEntries > 0) && (
              <Card className="lg:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Learning Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {routingIntelligence.affinity?.trackedPairs > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-2">Task Affinity ({routingIntelligence.affinity.trackedPairs} pairs)</h4>
                        <div className="space-y-1">
                          {routingIntelligence.affinity.topPairs?.slice(0, 5).map((pair, idx) => (
                            <div key={idx} className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">{pair.backend} + {pair.task}</span>
                              <span className={pair.score > 0.6 ? "text-green-500" : pair.score > 0.4 ? "text-yellow-500" : "text-red-500"}>
                                {(pair.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {routingIntelligence.prewarmStatus?.trackedEntries > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-2">Usage Patterns (Hour {routingIntelligence.prewarmStatus.currentHour})</h4>
                        {routingIntelligence.prewarmStatus.nextHourPredictions?.length > 0 ? (
                          <p className="text-xs text-muted-foreground">
                            Next hour: <span className="text-primary">{routingIntelligence.prewarmStatus.nextHourPredictions.join(", ")}</span>
                          </p>
                        ) : (
                          <p className="text-xs text-muted-foreground">Learning... ({routingIntelligence.prewarmStatus.trackedEntries} entries)</p>
                        )}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {activeTab === "logs" && (
          <div className="space-y-4">
            {/* Log Filter */}
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setSelectedLogProvider("all")}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                  selectedLogProvider === "all" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                }`}
              >
                All
              </button>
              {backendsResponse?.backends?.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setSelectedLogProvider(b.id)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                    selectedLogProvider === b.id ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                  }`}
                >
                  {b.name}
                </button>
              ))}
              <div className="ml-auto flex items-center gap-2">
                {isStreaming && <Badge variant="default" className="bg-green-500">Streaming</Badge>}
                <button onClick={() => setOllamaLogs([])} className="text-xs text-muted-foreground hover:text-foreground">
                  Clear
                </button>
              </div>
            </div>

            {/* Log Output */}
            <Card>
              <CardContent className="p-0">
                <div
                  className="rounded-lg p-4 font-mono text-xs h-[500px] overflow-y-auto"
                  style={{ backgroundColor: theme === 'dark' ? '#0D1A15' : '#1E3A32' }}
                >
                  {ollamaLogs.length > 0 ? (
                    <div className="space-y-0.5">
                      {ollamaLogs.map((log, idx) => (
                        <div
                          key={idx}
                          className={
                            log.includes('[THINK]') ? 'text-pink-400' :
                            log.includes('[ERROR]') ? 'text-red-400' :
                            log.includes('[STREAM]') ? 'text-amber-300' :
                            'text-slate-300'
                          }
                        >
                          <span className="text-slate-600 mr-2">{String(idx + 1).padStart(3, '0')}</span>
                          {log}
                        </div>
                      ))}
                      <div className="text-emerald-500 animate-pulse">▌</div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-500">
                      Waiting for logs...
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* Backend Modal */}
      {(showAddBackend || editingBackend) && (
        <BackendModal
          backend={editingBackend}
          onClose={() => { setShowAddBackend(false); setEditingBackend(null) }}
          onSave={async (data) => {
            try {
              const res = await fetch("/api/backends", {
                method: editingBackend ? "PUT" : "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
              })
              if (res.ok) {
                setShowAddBackend(false)
                setEditingBackend(null)
              }
            } catch { /* ignore */ }
          }}
          onDelete={editingBackend ? async (id) => {
            try {
              const res = await fetch(`/api/backends?id=${id}`, { method: "DELETE" })
              if (res.ok) setEditingBackend(null)
            } catch { /* ignore */ }
          } : undefined}
        />
      )}
    </div>
  )
}

function BackendModal({
  backend,
  onClose,
  onSave,
  onDelete
}: {
  backend: BackendStatus | null
  onClose: () => void
  onSave: (data: Partial<BackendStatus>) => Promise<void>
  onDelete?: (id: string) => Promise<void>
}) {
  const [formData, setFormData] = useState({
    id: backend?.id || "",
    name: backend?.name || "",
    provider: backend?.provider || "ollama" as const,
    type: backend?.type || "local" as const,
    url: backend?.url || "http://localhost:11434",
    enabled: backend?.enabled ?? true,
    priority: backend?.priority ?? 0,
    models: backend?.models || { quick: "", coder: "", moe: "", thinking: "" },
    api_key: backend?.api_key || ""
  })
  const [saving, setSaving] = useState(false)
  const [discoveredModels, setDiscoveredModels] = useState<{ name: string; sizeGB?: string }[]>([])
  const [loadingModels, setLoadingModels] = useState(false)

  const isGemini = formData.provider === "gemini"

  const pollModels = useCallback(async (url: string, provider: string, apiKey?: string) => {
    if (provider !== "gemini" && !url) return
    setLoadingModels(true)
    try {
      let fetchUrl = `/api/backends/models?provider=${provider}`
      if (provider !== "gemini") fetchUrl += `&url=${encodeURIComponent(url)}`
      if (apiKey) fetchUrl += `&apiKey=${encodeURIComponent(apiKey)}`
      const res = await fetch(fetchUrl)
      const data = await res.json()
      if (data.success && data.models) setDiscoveredModels(data.models)
      else setDiscoveredModels([])
    } catch {
      setDiscoveredModels([])
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (formData.provider === "gemini") pollModels("", formData.provider, formData.api_key)
      else if (formData.url) pollModels(formData.url, formData.provider)
    }, 500)
    return () => clearTimeout(timer)
  }, [formData.url, formData.provider, formData.api_key, pollModels])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    await onSave(formData)
    setSaving(false)
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-card border rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        <div className="p-4 border-b flex items-center justify-between">
          <h2 className="font-semibold">{backend ? "Edit Backend" : "Add Backend"}</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value, id: prev.id || e.target.value.toLowerCase().replace(/\s+/g, '-') }))}
              className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Provider</label>
              <select
                value={formData.provider}
                onChange={(e) => {
                  const provider = e.target.value as typeof formData.provider
                  const defaults: Record<string, { url: string; type: "local" | "remote" }> = {
                    ollama: { url: "http://localhost:11434", type: "local" },
                    llamacpp: { url: "http://localhost:8080", type: "local" },
                    lmstudio: { url: "http://localhost:1234", type: "local" },
                    vllm: { url: "http://localhost:8000", type: "local" },
                    gemini: { url: "https://generativelanguage.googleapis.com", type: "remote" },
                  }
                  const d = defaults[provider] || { url: formData.url, type: formData.type }
                  setFormData(prev => ({ ...prev, provider, url: d.url, type: d.type }))
                }}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
              >
                <option value="ollama">Ollama</option>
                <option value="llamacpp">llama.cpp</option>
                <option value="lmstudio">LM Studio</option>
                <option value="vllm">vLLM</option>
                <option value="openai">OpenAI-compatible</option>
                <option value="gemini">Google Gemini</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Type</label>
              <select
                value={formData.type}
                onChange={(e) => setFormData(prev => ({ ...prev, type: e.target.value as "local" | "remote" }))}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                disabled={isGemini}
              >
                <option value="local">Local</option>
                <option value="remote">Remote</option>
              </select>
            </div>
          </div>

          {isGemini ? (
            <div>
              <label className="block text-sm font-medium mb-1">API Key</label>
              <input
                type="password"
                value={formData.api_key}
                onChange={(e) => setFormData(prev => ({ ...prev, api_key: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm font-mono"
                placeholder="AIza..."
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium mb-1">URL</label>
              <input
                type="url"
                value={formData.url}
                onChange={(e) => setFormData(prev => ({ ...prev, url: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md bg-background text-sm"
                required
              />
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Models</label>
              {loadingModels && <span className="text-xs text-muted-foreground">Loading...</span>}
              {!loadingModels && discoveredModels.length > 0 && (
                <span className="text-xs text-primary">{discoveredModels.length} available</span>
              )}
            </div>
            <div className="space-y-2">
              {(["quick", "coder", "moe", "thinking"] as const).map((tier) => (
                <div key={tier} className="flex items-center gap-2">
                  <span className="w-16 text-xs text-muted-foreground capitalize">{tier}</span>
                  {discoveredModels.length > 0 ? (
                    <select
                      value={formData.models[tier]}
                      onChange={(e) => setFormData(prev => ({ ...prev, models: { ...prev.models, [tier]: e.target.value } }))}
                      className="flex-1 px-2 py-1 text-xs border rounded bg-background"
                    >
                      <option value="">Select...</option>
                      {discoveredModels.map((m) => (
                        <option key={m.name} value={m.name}>{m.name}{m.sizeGB ? ` (${m.sizeGB})` : ""}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={formData.models[tier]}
                      onChange={(e) => setFormData(prev => ({ ...prev, models: { ...prev.models, [tier]: e.target.value } }))}
                      className="flex-1 px-2 py-1 text-xs border rounded bg-background"
                      placeholder="model name"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={formData.enabled}
                onChange={(e) => setFormData(prev => ({ ...prev, enabled: e.target.checked }))}
              />
              Enabled
            </label>
            <div className="flex items-center gap-2">
              <label className="text-sm">Priority:</label>
              <input
                type="number"
                value={formData.priority}
                onChange={(e) => setFormData(prev => ({ ...prev, priority: parseInt(e.target.value) || 0 }))}
                className="w-16 px-2 py-1 text-sm border rounded bg-background"
                min={0}
              />
            </div>
          </div>

          <div className="flex items-center justify-between pt-4 border-t">
            {backend && onDelete ? (
              <button type="button" onClick={() => onDelete(backend.id)} className="text-sm text-destructive hover:underline">
                Delete
              </button>
            ) : <div />}
            <div className="flex gap-2">
              <button type="button" onClick={onClose} className="px-4 py-2 text-sm border rounded hover:bg-muted">
                Cancel
              </button>
              <button type="submit" disabled={saving} className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded disabled:opacity-50">
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
