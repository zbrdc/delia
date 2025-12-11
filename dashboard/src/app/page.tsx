"use client"

import { useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from "@/components/ui/chart"
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
  backend_type?: string  // "local" or "remote"
  backend?: string       // "ollama" or "llamacpp" (only if not default)
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

interface User {
  id: string
  email: string
  display_name?: string
  is_superuser: boolean
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
  provider: "ollama" | "llamacpp" | "vllm" | "openai" | "gemini" | "custom"
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

const PROVIDER_COLORS: Record<string, string> = {
  ollama: "#689B8A",
  llamacpp: "#4A7D6D",
  vllm: "#8BB5A6",
  openai: "#1E3A32",
  gemini: "#FF6B7A",
  custom: "#6B8F85",
}

const TYPE_COLORS: Record<string, string> = {
  local: "#689B8A",
  remote: "#FF6B7A",
}

const modelsConfig = {
  quick: { label: "Quick", color: "#8BB5A6" },
  coder: { label: "Coder", color: "#689B8A" },
  moe: { label: "MoE", color: "#FF6B7A" },
} satisfies ChartConfig

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
  ml_pipeline: "#689B8A",
  feature_pipeline: "#4A7D6D",
  other: "#6B8F85",
}

function WatermelonSeed({ className = "", style = {} }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg
      viewBox="0 0 10 16"
      className={className}
      style={style}
      fill="currentColor"
    >
      <ellipse cx="5" cy="8" rx="4" ry="7" />
    </svg>
  )
}

function ScatteredSeeds({ count = 5, className = "" }: { count?: number; className?: string }) {
  const seededRandom = (seed: number, offset: number) => {
    const x = Math.sin(seed * 9999 + offset * 7777) * 10000
    return x - Math.floor(x)
  }

  const seeds = Array.from({ length: count }, (_, i) => ({
    id: i,
    left: `${10 + (i * 18) + seededRandom(i, 1) * 12}%`,
    top: `${25 + seededRandom(i, 2) * 50}%`,
    rotation: -25 + seededRandom(i, 3) * 50,
    scale: 0.65 + seededRandom(i, 4) * 0.35,
    opacity: 0.08 + seededRandom(i, 5) * 0.06,
  }))

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {seeds.map((seed) => (
        <WatermelonSeed
          key={seed.id}
          className="absolute w-3 h-5 text-foreground"
          style={{
            left: seed.left,
            top: seed.top,
            transform: `rotate(${seed.rotation}deg) scale(${seed.scale})`,
            opacity: seed.opacity,
          }}
        />
      ))}
    </div>
  )
}

const normalizeModel = (model: string): "quick" | "coder" | "moe" | "thinking" => {
  const m = model.toLowerCase()
  if (m === "14b" || m === "quick") return "quick"
  if (m === "30b" || m === "coder") return "coder"
  if (m === "moe") return "moe"
  if (m === "thinking" || m.includes("think") || m.includes("reason")) return "thinking"
  return "quick" // default
}

const getModelColor = (model: string): string => {
  const tier = normalizeModel(model)
  return COLORS[tier]
}

export default function Dashboard() {
  const [stats, setStats] = useState<UsageStats | null>(null)
  const [enhanced, setEnhanced] = useState<EnhancedStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">("dark")
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<"overview" | "logs">("overview")
  const [ollamaLogs, setOllamaLogs] = useState<string[]>([])
  const [ollamaStatus, setOllamaStatus] = useState<{ model: string; size: number } | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [backendsResponse, setBackendsResponse] = useState<BackendsResponse | null>(null)
  const [activeBackend, setActiveBackend] = useState<string>("")
  const [secondsAgo, setSecondsAgo] = useState(0)
  const [expandedSections, setExpandedSections] = useState({
    usage: true,
    models: true,
    costs: false,
  })
  const [showAddBackend, setShowAddBackend] = useState(false)
  const [editingBackend, setEditingBackend] = useState<BackendStatus | null>(null)
  const [circuitBreaker, setCircuitBreaker] = useState<CircuitBreakerState | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [selectedLogProvider, setSelectedLogProvider] = useState<string>("all")

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

  const toggleTheme = () => {
    setTheme(prev => prev === "dark" ? "light" : "dark")
  }

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

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

  const fetchOllamaStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/ollama/stream?type=ps")
      if (res.ok) {
        const data = await res.json()
        if (data.models && data.models.length > 0) {
          const model = data.models[0]
          setOllamaStatus({ model: model.name, size: model.size })
        } else {
          setOllamaStatus(null)
        }
      }
    } catch {
      setOllamaStatus(null)
    }
  }, [])

  const fetchBackends = useCallback(async () => {
    try {
      const res = await fetch("/api/backends")
      if (res.ok) {
        const data = await res.json() as BackendsResponse
        if (data.backends) {
          setBackendsResponse(data)
          setActiveBackend(data.activeBackend || "")
        }
      }
    } catch {
    }
  }, [])

  const fetchCircuitBreaker = useCallback(async () => {
    try {
      const res = await fetch("/api/circuit-breaker")
      if (res.ok) {
        const data = await res.json()
        if (data.success) {
          setCircuitBreaker(data)
        }
      }
    } catch {
    }
  }, [])

  const fetchUser = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/me")
      if (res.ok) {
        const data = await res.json()
        setUser(data)
      } else {
        setUser(null)
      }
    } catch {
      setUser(null)
    }
  }, [])

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (selectedLogProvider === "all") {
        params.set("all", "true")
      } else {
        params.set("backend_id", selectedLogProvider)
      }
      const res = await fetch(`/api/logs?${params}`)
      if (res.ok) {
        const data = await res.json()
        if (data.logs && data.logs.length > 0) {
          const formattedLogs = data.logs.map((log: {
            type: string
            message: string
            provider?: string
            backend_id?: string
            backend?: string
            garden_msg?: string
            model?: string
            tokens?: number
          }) => {
            const displayMsg = log.garden_msg || log.message
            const backend = log.backend || log.backend_id || log.provider || ""
            const tokenInfo = log.tokens ? ` (${log.tokens} tokens)` : ""
            const modelInfo = log.model ? ` [${log.model}]` : ""
            if (backend) {
              return `[${backend}]${modelInfo} ${displayMsg}${tokenInfo}`
            }
            return `${displayMsg}${modelInfo}${tokenInfo}`
          })
          setOllamaLogs(formattedLogs)
          setIsStreaming(data.logs.some((log: { type: string }) => log.type === "STREAM"))
        } else {
          setOllamaLogs([])
          setIsStreaming(false)
        }
      }
    } catch {
    }
  }, [selectedLogProvider])

  useEffect(() => {
    fetchStats()
    fetchOllamaStatus()
    fetchBackends()
    fetchCircuitBreaker()
    fetchUser()
    const statsInterval = setInterval(() => {
      fetchStats()
      fetchOllamaStatus()
      fetchCircuitBreaker()
    }, 5000)
    const backendsInterval = setInterval(() => {
      fetchBackends()
    }, 3000)
    return () => {
      clearInterval(statsInterval)
      clearInterval(backendsInterval)
    }
  }, [fetchStats, fetchOllamaStatus, fetchBackends, fetchCircuitBreaker, fetchUser])

  useEffect(() => {
    if (activeTab === "logs") {
      fetchLogs()
      const interval = setInterval(fetchLogs, 1000)
      return () => clearInterval(interval)
    }
  }, [activeTab, fetchLogs])

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center relative">
        <ScatteredSeeds count={8} />
        <div className="flex flex-col items-center gap-4 z-10">
          <div className="relative">
            <div className="w-12 h-12 rounded-full border-4 border-primary border-t-accent animate-spin" />
            <div className="absolute inset-2 rounded-full bg-accent/20" />
          </div>
          <div className="text-muted-foreground">Loading dashboard...</div>
          <div className="flex gap-1 mt-2">
            <WatermelonSeed className="w-2 h-3 text-foreground/20 animate-bounce" style={{ animationDelay: "0ms" }} />
            <WatermelonSeed className="w-2 h-3 text-foreground/20 animate-bounce" style={{ animationDelay: "150ms" }} />
            <WatermelonSeed className="w-2 h-3 text-foreground/20 animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center relative">
        <ScatteredSeeds count={6} />
        <Card className="max-w-md relative z-10">
          <CardHeader>
            <CardTitle className="text-destructive flex items-center gap-2">
              <WatermelonSeed className="w-3 h-4 rotate-45" />
              Connection Error
            </CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Make sure the Delia server is running and the .usage_stats.json file exists.
            </p>
            <button
              onClick={fetchStats}
              className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
              Retry
            </button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const totalCalls = (stats?.quick?.calls || 0) + (stats?.coder?.calls || 0) + (stats?.moe?.calls || 0) + (stats?.thinking?.calls || 0)
  const totalTokens = (stats?.quick?.tokens || 0) + (stats?.coder?.tokens || 0) + (stats?.moe?.tokens || 0) + (stats?.thinking?.tokens || 0)
  
  const efficiencyQuick = totalCalls > 0 ? ((stats?.quick?.calls || 0) / totalCalls * 100) : 0

  const usageBarData = [
    { name: "Quick", calls: stats?.quick?.calls || 0, tokens: stats?.quick?.tokens || 0, color: COLORS.quick, desc: "Fast Q&A" },
    { name: "Coder", calls: stats?.coder?.calls || 0, tokens: stats?.coder?.tokens || 0, color: COLORS.coder, desc: "Code tasks" },
    { name: "MoE", calls: stats?.moe?.calls || 0, tokens: stats?.moe?.tokens || 0, color: COLORS.moe, desc: "Complex reasoning" },
  ]

  const backendStats = (enhanced?.recent_calls || []).reduce((acc, call) => {
    const type = (call.backend_type || "local") as "local" | "remote"
    acc[type].calls += 1
    acc[type].tokens += call.tokens || 0
    acc[type].totalMs += call.elapsed_ms || 0
    return acc
  }, { 
    local: { calls: 0, tokens: 0, totalMs: 0 }, 
    remote: { calls: 0, tokens: 0, totalMs: 0 }
  })

  const localAvgMs = backendStats.local.calls > 0 ? Math.round(backendStats.local.totalMs / backendStats.local.calls) : 0
  const remoteAvgMs = backendStats.remote.calls > 0 ? Math.round(backendStats.remote.totalMs / backendStats.remote.calls) : 0

  const avgResponseTime = enhanced?.recent_calls?.length 
    ? Math.round(enhanced.recent_calls.reduce((sum, c) => sum + c.elapsed_ms, 0) / enhanced.recent_calls.length)
    : 0

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/logo.svg" alt="Delia" className="w-8 h-8" />
              <span className="font-semibold text-lg hidden sm:inline">Delia</span>
              <span className="text-muted-foreground text-sm hidden md:inline">— Local LLM Delegation</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            
            <div className="flex items-center gap-3 mr-2">
              <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-muted/50">
                {backendsResponse?.backends?.slice(0, 4).map((backend) => (
                  <div 
                    key={backend.id}
                    className="flex items-center gap-1" 
                    title={`${backend.name}: ${backend.health.available ? 'online' : 'offline'}${backend.health.loaded_models?.length ? ` (${backend.health.loaded_models.length} models)` : ''}`}
                  >
                    <div 
                      className={`w-2 h-2 rounded-full ${
                        backend.health.available 
                          ? activeBackend === backend.id ? 'animate-pulse' : '' 
                          : 'opacity-30'
                      }`}
                      style={{ backgroundColor: PROVIDER_COLORS[backend.provider] || PROVIDER_COLORS.custom }}
                    />
                    <span className={`text-xs ${activeBackend === backend.id ? 'font-medium text-foreground' : 'text-muted-foreground'}`}>
                      {backend.type === 'local' ? 'L' : 'R'}
                    </span>
                  </div>
                ))}
                {backendsResponse?.summary && backendsResponse.summary.total > 4 && (
                  <span className="text-xs text-muted-foreground">+{backendsResponse.summary.total - 4}</span>
                )}
              </div>
              
              <span className="text-xs text-muted-foreground/60 hidden lg:inline">
                · {secondsAgo}s ago
              </span>
            </div>

            
            <div className="hidden md:flex gap-1.5">
              <Badge variant="outline" className="text-xs" style={{ backgroundColor: '#8BB5A620', color: '#8BB5A6', borderColor: '#8BB5A650' }}>
                Quick: {stats?.quick?.calls || 0}
              </Badge>
              <Badge variant="outline" className="text-xs" style={{ backgroundColor: '#689B8A20', color: '#689B8A', borderColor: '#689B8A50' }}>
                Coder: {stats?.coder?.calls || 0}
              </Badge>
              <Badge variant="outline" className="text-xs" style={{ backgroundColor: '#FF6B7A20', color: '#FF6B7A', borderColor: '#FF6B7A50' }}>
                MoE: {stats?.moe?.calls || 0}
              </Badge>
              <Badge
                variant="outline"
                className="text-xs"
                style={{
                  backgroundColor: efficiencyQuick >= 70 ? '#689B8A20' : efficiencyQuick >= 40 ? '#8BB5A620' : '#FF6B7A20',
                  color: efficiencyQuick >= 70 ? '#689B8A' : efficiencyQuick >= 40 ? '#8BB5A6' : '#FF6B7A',
                  borderColor: efficiencyQuick >= 70 ? '#689B8A50' : efficiencyQuick >= 40 ? '#8BB5A650' : '#FF6B7A50'
                }}
              >
                {efficiencyQuick.toFixed(0)}% quick
              </Badge>
            </div>

            
            <button
              onClick={toggleTheme}
              className="p-2 rounded-md hover:bg-accent transition-colors"
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>

            
            <div className="relative">
              <button
                onClick={() => setMenuOpen(!menuOpen)}
                className="p-2 rounded-md hover:bg-accent transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              {menuOpen && (
                <div className="absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-card border border-border">
                  <div className="py-1">
                    <button
                      onClick={() => { setActiveTab("overview"); setMenuOpen(false) }}
                      className={`block w-full text-left px-4 py-2 text-sm hover:bg-accent ${activeTab === "overview" ? "text-primary font-medium" : ""}`}
                    >
                      📊 Overview
                    </button>
                    <button
                      onClick={() => { setActiveTab("logs"); setMenuOpen(false) }}
                      className={`block w-full text-left px-4 py-2 text-sm hover:bg-accent ${activeTab === "logs" ? "text-primary font-medium" : ""}`}
                    >
                      🖥️ Live Ollama
                    </button>
                    <hr className="my-1 border-border" />
                    <button
                      onClick={fetchStats}
                      className="block w-full text-left px-4 py-2 text-sm hover:bg-accent"
                    >
                      🔄 Refresh
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      
      <main className="container mx-auto px-4 py-6">
        {activeTab === "overview" && (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
            
            <div className="space-y-4">
              
              <Card>
                <CardContent className="py-3">
                  <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                    
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <WatermelonSeed className="w-3 h-4 text-primary" />
                        <span className="text-lg font-bold">{totalCalls.toLocaleString()}</span>
                        <span className="text-sm font-medium text-muted-foreground">calls</span>
                      </div>
                      <div className="w-px h-4 bg-border" />
                      <div className="flex items-center gap-1.5">
                        <span className="text-lg font-bold">{totalTokens >= 1000000 ? `${(totalTokens / 1000000).toFixed(1)}M` : totalTokens >= 1000 ? `${(totalTokens / 1000).toFixed(0)}k` : totalTokens}</span>
                        <span className="text-sm font-medium text-muted-foreground">tokens</span>
                      </div>
                    </div>

                    
                    <div className="flex items-center gap-3">
                      {usageBarData.map((item) => (
                        <div key={item.name} className="flex items-center gap-1" title={`${item.name}: ${item.tokens.toLocaleString()} tokens`}>
                          <WatermelonSeed className="w-2.5 h-4" style={{ color: item.color }} />
                          <span className="text-xs font-medium text-muted-foreground">{item.tokens >= 1000 ? `${(item.tokens/1000).toFixed(0)}k` : item.tokens || '-'}</span>
                        </div>
                      ))}
                    </div>

                    
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-medium flex items-center gap-1" title={`Local: ${backendStats.local.tokens.toLocaleString()} tokens`}>
                        <span style={{ color: '#689B8A' }}>🌱</span>
                        <span className="text-muted-foreground">{backendStats.local.tokens >= 1000 ? `${(backendStats.local.tokens/1000).toFixed(0)}k` : backendStats.local.tokens}</span>
                      </span>
                      <span className="text-xs font-medium flex items-center gap-1" title={`Remote: ${backendStats.remote.tokens.toLocaleString()} tokens`}>
                        <span style={{ color: '#FF6B7A' }}>☁️</span>
                        <span className="text-muted-foreground">{backendStats.remote.tokens >= 1000 ? `${(backendStats.remote.tokens/1000).toFixed(0)}k` : backendStats.remote.tokens}</span>
                      </span>
                    </div>

                    
                    <div className="flex items-center gap-2 ml-auto">
                      <span className="text-xs font-medium text-muted-foreground">avg</span>
                      <span className="text-sm font-semibold" style={{ color: '#689B8A' }}>
                        {localAvgMs >= 1000 ? `${(localAvgMs / 1000).toFixed(1)}s` : `${localAvgMs}ms`}
                      </span>
                      <span className="text-xs font-medium text-muted-foreground">/</span>
                      <span className="text-sm font-semibold" style={{ color: '#FF6B7A' }}>
                        {remoteAvgMs >= 1000 ? `${(remoteAvgMs / 1000).toFixed(1)}s` : `${remoteAvgMs}ms`}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              
              <Card className="lg:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <WatermelonSeed className="w-3 h-4 text-primary" />
                    Recent Harvests
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {enhanced?.recent_calls && enhanced.recent_calls.length > 0 ? (
                    <div className="space-y-1.5 max-h-[320px] overflow-y-auto">
                      {[...enhanced.recent_calls].reverse().slice(0, 15).map((call, idx) => {
                        const tier = normalizeModel(call.model)
                        const tierLabels = { quick: "Quick", coder: "Coder", moe: "MoE", thinking: "Thinking" }
                        const backendType = call.backend_type || "local"
                        const isRemote = backendType === "remote"
                        const rowBg = idx % 2 === 0
                          ? "bg-[#689B8A]/10 dark:bg-[#A8D4C4]/10"
                          : "bg-[#FF6B7A]/8 dark:bg-[#FF8A95]/10"
                        return (
                        <div key={idx} className={`flex items-start gap-3 py-2 px-3 rounded-md ${rowBg} hover:bg-accent/20 transition-colors`}>
                          
                          <WatermelonSeed
                            className="w-3 h-5 mt-0.5 flex-shrink-0"
                            style={{ color: COLORS[tier] }}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <Badge variant="outline" className="text-xs font-medium px-1.5 py-0.5" style={{ borderColor: COLORS[tier], color: COLORS[tier] }}>
                                {tierLabels[tier]}
                              </Badge>
                              <Badge variant="secondary" className="text-xs font-medium px-1.5 py-0.5" style={{ backgroundColor: `${TASK_COLORS[call.task_type] || TASK_COLORS.other}20`, color: TASK_COLORS[call.task_type] || TASK_COLORS.other }}>
                                {call.task_type}
                              </Badge>
                              
                              <span
                                className="text-xs font-medium px-1.5 rounded"
                                style={{
                                  color: isRemote ? "#FF8A95" : "#689B8A",
                                  backgroundColor: isRemote ? "#FF6B7A15" : "#689B8A15"
                                }}
                              >
                                {isRemote ? "☁️" : "🌱"}
                              </span>
                              <span className="text-xs font-medium text-muted-foreground ml-auto">{call.tokens.toLocaleString()} tokens · {(call.elapsed_ms / 1000).toFixed(1)}s</span>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1 truncate">{call.preview}</p>
                          </div>
                        </div>
                      )})}
                    </div>
                  ) : (
                    <div className="h-[100px] flex flex-col items-center justify-center text-muted-foreground text-sm relative">
                      <div className="flex gap-2 mb-2">
                        <WatermelonSeed className="w-3 h-4 rotate-[-20deg] opacity-30" />
                        <WatermelonSeed className="w-2 h-3 rotate-[15deg] opacity-20" />
                        <WatermelonSeed className="w-3 h-4 rotate-[30deg] opacity-30" />
                      </div>
                      No melons harvested yet...
                    </div>
                  )}
                </CardContent>
              </Card>

              {user && (
                <Card className="mb-6">
                  <CardContent className="pt-4">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Logged in as {user.display_name || user.email}</span>
                      {user.is_superuser && <Badge variant="outline">Superuser</Badge>}
                    </div>
                  </CardContent>
                </Card>
              )}

              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Task Types</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {enhanced?.task_stats && Object.values(enhanced.task_stats).some(v => v > 0) ? (
                      <div className="h-[180px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart 
                            data={Object.entries(enhanced.task_stats)
                              .filter(([, v]) => v > 0)
                              .sort((a, b) => b[1] - a[1])
                              .map(([name, value]) => ({ name, value, fill: TASK_COLORS[name] || TASK_COLORS.other }))}
                            layout="vertical"
                            margin={{ top: 5, right: 30, bottom: 5, left: 60 }}
                          >
                            <XAxis type="number" tick={{ fontSize: 10 }} />
                            <YAxis 
                              type="category" 
                              dataKey="name" 
                              tick={{ fontSize: 10 }} 
                              width={55}
                            />
                            <Tooltip 
                              formatter={(value: number) => [value, "calls"]}
                              contentStyle={{ fontSize: 12 }}
                            />
                            <Bar 
                              dataKey="value" 
                              radius={[0, 4, 4, 0]}
                              label={{ position: 'right', fontSize: 10, fill: '#888' }}
                            >
                              {Object.entries(enhanced.task_stats)
                                .filter(([, v]) => v > 0)
                                .sort((a, b) => b[1] - a[1])
                                .map(([name], index) => (
                                  <Cell key={`cell-${index}`} fill={TASK_COLORS[name] || TASK_COLORS.other} />
                                ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-[180px] flex flex-col items-center justify-center text-muted-foreground text-xs">
                        <div className="flex gap-1.5 mb-2">
                          <WatermelonSeed className="w-2 h-3 rotate-[-15deg] opacity-25" />
                          <WatermelonSeed className="w-2.5 h-3.5 rotate-[20deg] opacity-20" />
                        </div>
                        No task data yet
                      </div>
                    )}
                  </CardContent>
                </Card>

                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Response Times</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {enhanced?.response_times && (enhanced.response_times.quick?.length > 0 || enhanced.response_times.coder?.length > 0 || enhanced.response_times.moe?.length > 0 || enhanced.response_times.thinking?.length > 0) ? (
                      <div className="h-[180px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={(() => {
                            const quickData = (enhanced.response_times.quick || []).slice(-15);
                            const coderData = (enhanced.response_times.coder || []).slice(-15);
                            const moeData = (enhanced.response_times.moe || []).slice(-15);
                            const thinkingData = (enhanced.response_times.thinking || []).slice(-15);
                            const maxLen = Math.max(quickData.length, coderData.length, moeData.length, thinkingData.length);
                            const result: Array<{ idx: number; quick?: number; coder?: number; moe?: number; thinking?: number }> = [];
                            for (let i = 0; i < maxLen; i++) {
                              result.push({
                                idx: i,
                                quick: quickData[i]?.ms,
                                coder: coderData[i]?.ms,
                                moe: moeData[i]?.ms,
                                thinking: thinkingData[i]?.ms,
                              });
                            }
                            return result;
                          })()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="idx" tick={false} />
                            <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}s`} width={30} />
                            <Tooltip formatter={(value: number) => [`${value.toLocaleString()}ms`, ""]} />
                            <Line type="monotone" dataKey="quick" stroke={COLORS.quick} strokeWidth={2} dot={false} connectNulls />
                            <Line type="monotone" dataKey="coder" stroke={COLORS.coder} strokeWidth={2} dot={false} connectNulls />
                            <Line type="monotone" dataKey="moe" stroke={COLORS.moe} strokeWidth={2} dot={false} connectNulls />
                            <Line type="monotone" dataKey="thinking" stroke={COLORS.thinking} strokeWidth={2} dot={false} connectNulls />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-[180px] flex flex-col items-center justify-center text-muted-foreground text-sm">
                        <div className="flex gap-1.5 mb-2">
                          <WatermelonSeed className="w-2 h-3 rotate-[10deg] opacity-25" />
                          <WatermelonSeed className="w-2.5 h-3.5 rotate-[-25deg] opacity-20" />
                        </div>
                        No response data yet
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>

            
            <div className="space-y-4">
              
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-base flex items-center gap-2">
                        <WatermelonSeed className="w-3 h-4 rotate-[-15deg]" />
                        Gardens
                      </CardTitle>
                      <CardDescription className="text-xs">
                        {backendsResponse?.summary
                          ? `${backendsResponse.summary.available}/${backendsResponse.summary.enabled} gardens thriving`
                          : "Checking the patch..."}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setShowAddBackend(true)}
                        className="text-xs px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 text-primary transition-colors flex items-center gap-1"
                      >
                        <WatermelonSeed className="w-2 h-3" />
                        Add
                      </button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2">
                  {backendsResponse?.backends?.map((backend) => {
                    const isActive = activeBackend === backend.id
                    const isHealthy = backend.health.available
                    return (
                      <div
                        key={backend.id}
                        className={`p-2.5 rounded-lg border cursor-pointer transition-colors ${
                          isHealthy
                            ? isActive
                              ? "border-accent/50 bg-accent/5"
                              : "border-primary/30 bg-primary/5 hover:border-primary/50"
                            : "border-border bg-muted/30"
                        }`}
                        onClick={() => setEditingBackend(backend)}
                      >
                        
                        <div className="flex items-center gap-2">
                          <WatermelonSeed
                            className={`w-3 h-4 flex-shrink-0 ${isHealthy ? "" : "opacity-40"}`}
                            style={{ color: isActive ? '#FF6B7A' : (PROVIDER_COLORS[backend.provider] || PROVIDER_COLORS.custom) }}
                          />
                          <span className="font-medium text-sm flex-1 truncate">{backend.name}</span>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded-full"
                            style={{
                              backgroundColor: isHealthy ? (isActive ? '#FF6B7A20' : '#689B8A20') : 'transparent',
                              color: isHealthy ? (isActive ? '#FF6B7A' : '#689B8A') : 'inherit'
                            }}
                          >
                            {backend.health.circuit_open ? '🥀 drought' : isActive ? '🌱 active' : isHealthy ? '✓ ready' : 'dormant'}
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-2 mt-1.5 text-xs text-muted-foreground">
                          <span
                            className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                            style={{
                              backgroundColor: `${TYPE_COLORS[backend.type]}15`,
                              color: TYPE_COLORS[backend.type]
                            }}
                          >
                            {backend.type}
                          </span>
                          <span className="truncate flex-1">{backend.url}</span>
                        </div>
                        
                        {(backend.health.response_time_ms > 0 || backend.health.loaded_models?.length > 0) && (
                          <div className="flex items-center gap-3 mt-1 text-[10px] text-muted-foreground">
                            {backend.health.response_time_ms > 0 && (
                              <span>{backend.health.response_time_ms}ms</span>
                            )}
                            {backend.health.loaded_models?.length > 0 && (
                              <span>{backend.health.loaded_models.length} models</span>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                  
                  {(!backendsResponse?.backends || backendsResponse.backends.length === 0) && (
                    <div className="text-center py-6 text-muted-foreground">
                      <div className="flex justify-center gap-3 mb-3">
                        <WatermelonSeed className="w-4 h-6 rotate-[-20deg] opacity-20" />
                        <WatermelonSeed className="w-5 h-7 rotate-[5deg] opacity-30" />
                        <WatermelonSeed className="w-4 h-6 rotate-[25deg] opacity-20" />
                      </div>
                      <p className="text-sm font-medium mb-1">No gardens configured</p>
                      <p className="text-xs opacity-70">Add a garden to start growing melons</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              
              {circuitBreaker && Object.values(circuitBreaker).some((status: CircuitBreakerStatus) => status.circuit_open) && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Circuit Breaker</CardTitle>
                    <CardDescription className="text-xs">
                      Backend failure protection and recovery
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {Object.entries(circuitBreaker).filter(([key]) => key !== 'active_backend' && key !== 'timestamp').map(([backend, status]: [string, CircuitBreakerStatus]) => (
                      status.circuit_open && (
                        <div key={backend} className="p-2.5 rounded-lg border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/20">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium text-sm capitalize">{backend}</span>
                            <Badge variant="destructive" className="text-xs">
                              OPEN
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground space-y-0.5">
                            <div>Fails: {status.consecutive_failures}</div>
                            {status.last_error && <div>Error: {status.last_error}</div>}
                            {status.seconds_until_available > 0 && (
                              <div>Recovery: {Math.ceil(status.seconds_until_available)}s</div>
                            )}
                            <div>Safe Context: {status.safe_context_kb}KB</div>
                          </div>
                        </div>
                      )
                    ))}
                  </CardContent>
                </Card>
              )}

              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Routing Strategy</CardTitle>
                  <CardDescription className="text-xs">
                    How requests are distributed across backends
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Routing Mode</label>
                    {(() => {
                      const currentRoutingMode = backendsResponse?.routing?.load_balance ? "load_balance" : backendsResponse?.routing?.prefer_local ? "prefer_local" : "prefer_remote"
                      const routingLabels = {
                        prefer_local: "Prioritize Local GPUs",
                        prefer_remote: "Prioritize Remote GPUs",
                        load_balance: "Load Balance"
                      }
                      return (
                        <Select
                          value={currentRoutingMode}
                          onValueChange={async (value: string) => {
                            const updates = {
                              prefer_local: value === "prefer_local",
                              load_balance: value === "load_balance"
                            }
                            try {
                              const res = await fetch("/api/backends", {
                                method: "PATCH",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify(updates)
                              })
                              if (res.ok) {
                                setBackendsResponse(prev => prev ? {
                                  ...prev,
                                  routing: { ...prev.routing, ...updates }
                                } : null)
                              }
                            } catch (e) { console.error(e) }
                          }}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue>{routingLabels[currentRoutingMode]}</SelectValue>
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="prefer_local">
                              <div>
                                <div className="font-medium">Prioritize Local GPUs</div>
                                <div className="text-xs text-muted-foreground">Use local GPU first, remote only if unavailable</div>
                              </div>
                            </SelectItem>
                            <SelectItem value="prefer_remote">
                              <div>
                                <div className="font-medium">Prioritize Remote GPUs</div>
                                <div className="text-xs text-muted-foreground">Use remote GPU first, local as backup</div>
                              </div>
                            </SelectItem>
                            <SelectItem value="load_balance">
                              <div>
                                <div className="font-medium">Load Balance</div>
                                <div className="text-xs text-muted-foreground">Distribute across all backends by priority weight</div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      )
                    })()}
                  </div>
                  
                  
                  <div className="pt-2 border-t">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium">Auto-Fallback</span>
                        <p className="text-xs text-muted-foreground">Try next backend if primary fails</p>
                      </div>
                      <button
                        onClick={async () => {
                          const newValue = !backendsResponse?.routing?.fallback_enabled
                          try {
                            const res = await fetch("/api/backends", {
                              method: "PATCH",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ fallback_enabled: newValue })
                            })
                            if (res.ok) {
                              setBackendsResponse(prev => prev ? {
                                ...prev,
                                routing: { ...prev.routing, fallback_enabled: newValue }
                              } : null)
                            }
                          } catch (e) { console.error(e) }
                        }}
                        className={`w-10 h-5 rounded-full transition-colors relative ${
                          backendsResponse?.routing?.fallback_enabled ? "bg-primary" : "bg-muted"
                        }`}
                      >
                        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${
                          backendsResponse?.routing?.fallback_enabled ? "left-5" : "left-0.5"
                        }`} />
                      </button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === "logs" && (
          <div className="space-y-4">
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <WatermelonSeed
                      className={`w-2.5 h-3.5 rotate-[-10deg] ${ollamaStatus ? '' : 'opacity-40'}`}
                      style={{ color: ollamaStatus ? '#689B8A' : undefined }}
                    />
                    Active Vine
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {ollamaStatus ? (
                    <div className="space-y-1">
                      <div className="font-medium" style={{ color: '#689B8A' }}>{ollamaStatus.model}</div>
                      <div className="text-xs text-muted-foreground">
                        {(ollamaStatus.size / 1024 / 1024 / 1024).toFixed(1)} GB loaded
                      </div>
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm flex items-center gap-2">
                      <WatermelonSeed className="w-2 h-3 opacity-30" />
                      No vine growing
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <WatermelonSeed className="w-2.5 h-3.5 rotate-[10deg]" style={{ color: isStreaming ? '#FF6B7A' : '#689B8A' }} />
                    Stream Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${isStreaming ? 'animate-pulse' : ''}`}
                      style={{ backgroundColor: isStreaming ? '#FF6B7A' : '#6B8F85' }}
                    />
                    <span className="text-sm">{isStreaming ? 'Streaming...' : 'Idle'}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Log Lines</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold" style={{ color: '#689B8A' }}>{ollamaLogs.length}</div>
                  <button
                    onClick={() => setOllamaLogs([])}
                    className="text-xs text-muted-foreground hover:text-accent transition-colors"
                  >
                    Clear logs
                  </button>
                </CardContent>
              </Card>
            </div>

            
            <Card className="mb-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <WatermelonSeed className="w-3 h-4 rotate-[-10deg]" />
                  Select Garden
                </CardTitle>
                <CardDescription className="text-xs">Choose which garden&apos;s logs to view</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedLogProvider("all")}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-1.5 ${
                      selectedLogProvider === "all"
                        ? "text-white"
                        : "bg-muted hover:bg-muted/80 text-muted-foreground"
                    }`}
                    style={selectedLogProvider === "all" ? { backgroundColor: '#689B8A' } : {}}
                  >
                    <WatermelonSeed className="w-2 h-3" />
                    All Gardens
                  </button>
                  {backendsResponse?.backends?.map((backend) => (
                    <button
                      key={backend.id}
                      onClick={() => setSelectedLogProvider(backend.id)}
                      className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-1.5 ${
                        selectedLogProvider === backend.id
                          ? "text-white"
                          : "bg-muted hover:bg-muted/80 text-muted-foreground"
                      }`}
                      style={selectedLogProvider === backend.id ? {
                        backgroundColor: backend.type === "local" ? '#689B8A' : '#FF6B7A'
                      } : {}}
                    >
                      <WatermelonSeed
                        className="w-2 h-3"
                        style={{ color: selectedLogProvider === backend.id ? 'white' : (PROVIDER_COLORS[backend.provider] || '#6B8F85') }}
                      />
                      {backend.name}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <WatermelonSeed className="w-3 h-4 rotate-[-5deg]" />
                    Live Output
                  </span>
                  <Badge
                    variant={isStreaming ? "default" : "secondary"}
                    style={isStreaming ? { backgroundColor: '#FF6B7A' } : {}}
                  >
                    {isStreaming ? "🍉 RIPENING" : "💤 DORMANT"}
                  </Badge>
                </CardTitle>
                <CardDescription>
                  Real-time output from your gardens when delegations occur
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="rounded-lg p-4 font-mono text-sm h-[400px] overflow-y-auto"
                  style={{
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    backgroundColor: theme === 'dark' ? '#0D1A15' : '#1E3A32',
                  }}
                >
                  {ollamaLogs.length > 0 ? (
                    <div className="space-y-1">
                      {ollamaLogs.map((log, idx) => (
                        <div
                          key={idx}
                          className={`${
                            log.includes('[THINK]') ? 'text-pink-400' :
                            log.includes('[MODEL]') ? 'text-emerald-400' :
                            log.includes('[ERROR]') ? 'text-red-400' :
                            log.includes('[INFO]') ? 'text-teal-300' :
                            log.includes('[STREAM]') ? 'text-amber-300' :
                            'text-slate-300'
                          }`}
                        >
                          <span className="text-slate-600 mr-2">{String(idx + 1).padStart(3, '0')}</span>
                          {log}
                        </div>
                      ))}
                      <div className="text-emerald-500 animate-pulse">▌</div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-500">
                      <div className="text-center">
                        <div className="flex justify-center gap-2 mb-3">
                          <WatermelonSeed className="w-3 h-4 rotate-[-15deg] text-slate-600" />
                          <WatermelonSeed className="w-2.5 h-3.5 rotate-[20deg] text-slate-700" />
                          <WatermelonSeed className="w-3 h-4 rotate-[5deg] text-slate-600" />
                        </div>
                        <p className="text-slate-400">Waiting for vines to grow...</p>
                        <p className="text-xs mt-2 text-slate-500">Logs will appear here when delegations run</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      
      <footer className="border-t bg-card mt-auto relative overflow-hidden">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between text-xs text-muted-foreground relative z-10">
          <span className="flex items-center gap-2">
            <WatermelonSeed className="w-2 h-3 rotate-[-20deg] opacity-40" />
            Delia Dashboard
          </span>
          <span className="flex items-center gap-2">
            {totalCalls} total calls • {totalTokens.toLocaleString()} tokens
            <WatermelonSeed className="w-2 h-3 rotate-[15deg] opacity-40" />
          </span>
        </div>
      </footer>

      
      {menuOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setMenuOpen(false)}
        />
      )}

      
      {(showAddBackend || editingBackend) && (
        <BackendModal
          backend={editingBackend}
          onClose={() => {
            setShowAddBackend(false)
            setEditingBackend(null)
          }}
          onSave={async (backendData) => {
            try {
              const isEdit = !!editingBackend
              const res = await fetch("/api/backends", {
                method: isEdit ? "PUT" : "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(backendData)
              })
              if (res.ok) {
                setShowAddBackend(false)
                setEditingBackend(null)
                const backendsRes = await fetch("/api/backends")
                if (backendsRes.ok) {
                  const data = await backendsRes.json()
                  setBackendsResponse(data)
                }
              }
            } catch (err) {
              console.error("Failed to save backend:", err)
            }
          }}
          onDelete={editingBackend ? async (id) => {
            try {
              const res = await fetch(`/api/backends?id=${id}`, {
                method: "DELETE"
              })
              if (res.ok) {
                setEditingBackend(null)
                const backendsRes = await fetch("/api/backends")
                if (backendsRes.ok) {
                  const data = await backendsRes.json()
                  setBackendsResponse(data)
                }
              }
            } catch (err) {
              console.error("Failed to delete backend:", err)
            }
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
  const [discoveredModels, setDiscoveredModels] = useState<{name: string, sizeGB?: string, description?: string}[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)

  const isGemini = formData.provider === "gemini"

  const pollModels = useCallback(async (url: string, provider: string, apiKey?: string) => {
    if (provider !== "gemini" && !url) return
    setLoadingModels(true)
    setModelError(null)
    try {
      let fetchUrl = `/api/backends/models?provider=${provider}`
      if (provider !== "gemini") {
        fetchUrl += `&url=${encodeURIComponent(url)}`
      }
      if (apiKey) {
        fetchUrl += `&apiKey=${encodeURIComponent(apiKey)}`
      }
      const res = await fetch(fetchUrl)
      const data = await res.json()
      if (data.success && data.models) {
        setDiscoveredModels(data.models)
      } else {
        setModelError(data.error || "Failed to fetch models")
        setDiscoveredModels([])
      }
    } catch (err) {
      setModelError(err instanceof Error ? err.message : "Connection failed")
      setDiscoveredModels([])
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (formData.provider === "gemini") {
        pollModels("", formData.provider, formData.api_key)
      } else if (formData.url) {
        pollModels(formData.url, formData.provider)
      }
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
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <WatermelonSeed className="w-3 h-4 rotate-[-10deg]" />
            {backend ? "Edit Garden" : "Add New Garden"}
          </h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          
          <div>
            <label className="block text-sm font-medium mb-1">Garden Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value, id: prev.id || e.target.value.toLowerCase().replace(/\s+/g, '-') }))}
              className="w-full px-3 py-2 border rounded-md bg-background"
              placeholder="My Local GPU"
              required
            />
          </div>

          
          <div>
            <label className="block text-sm font-medium mb-1">Provider</label>
            <select
              value={formData.provider}
              onChange={(e) => {
                const provider = e.target.value as typeof formData.provider
                let url = formData.url
                let type = formData.type
                if (provider === "gemini") {
                  url = "https://generativelanguage.googleapis.com"
                  type = "remote"
                } else if (provider === "ollama") {
                  url = "http://localhost:11434"
                  type = "local"
                } else if (provider === "llamacpp") {
                  url = "http://localhost:8080"
                } else if (provider === "vllm") {
                  url = "http://localhost:8000"
                }
                setFormData(prev => ({ ...prev, provider, url, type }))
              }}
              className="w-full px-3 py-2 border rounded-md bg-background"
            >
              <option value="ollama">Ollama</option>
              <option value="llamacpp">llama.cpp</option>
              <option value="vllm">vLLM</option>
              <option value="openai">OpenAI-compatible</option>
              <option value="gemini">Google Gemini</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          
          {!isGemini && (
            <div>
              <label className="block text-sm font-medium mb-1">Type</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="type"
                    value="local"
                    checked={formData.type === "local"}
                    onChange={() => {
                      setFormData(prev => ({
                        ...prev,
                        type: "local",
                        provider: "ollama",
                        url: "http://localhost:11434"
                      }))
                    }}
                  />
                  <span>Local</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="type"
                    value="remote"
                    checked={formData.type === "remote"}
                    onChange={() => {
                      setFormData(prev => ({
                        ...prev,
                        type: "remote",
                        provider: "llamacpp",
                        url: "http://localhost:8080"
                      }))
                    }}
                  />
                  <span>Remote</span>
                </label>
              </div>
            </div>
          )}

          
          {isGemini && (
            <div>
              <label className="block text-sm font-medium mb-1">
                API Key
                <a
                  href="https://aistudio.google.com/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 text-xs text-primary hover:underline"
                >
                  Get API key →
                </a>
              </label>
              <input
                type="password"
                value={formData.api_key}
                onChange={(e) => setFormData(prev => ({ ...prev, api_key: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md bg-background font-mono text-sm"
                placeholder="AIza..."
              />
              <p className="text-xs text-muted-foreground mt-1">
                Free tier: 15 requests/minute, 1M tokens/day
              </p>
            </div>
          )}

          
          {!isGemini && (
            <div>
              <label className="block text-sm font-medium mb-1">URL</label>
              <input
                type="url"
                value={formData.url}
                onChange={(e) => setFormData(prev => ({ ...prev, url: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md bg-background"
                placeholder="http://localhost:11434"
                required
              />
            </div>
          )}

          
          <div>
            <label className="block text-sm font-medium mb-1">Priority (lower = higher priority)</label>
            <input
              type="number"
              value={formData.priority}
              onChange={(e) => setFormData(prev => ({ ...prev, priority: parseInt(e.target.value) || 0 }))}
              className="w-full px-3 py-2 border rounded-md bg-background"
              min={0}
              max={100}
            />
          </div>

          
          <div>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={formData.enabled}
                onChange={(e) => setFormData(prev => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="text-sm font-medium">Enabled</span>
            </label>
          </div>

          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium">Vine Configuration</label>
              {loadingModels && <span className="text-xs text-muted-foreground">Loading models...</span>}
              {!loadingModels && discoveredModels.length > 0 && (
                <span className="text-xs text-primary">{discoveredModels.length} models available</span>
              )}
              {!loadingModels && modelError && !isGemini && (
                <span className="text-xs text-destructive">{modelError}</span>
              )}
            </div>
            <div className="space-y-2">
              {(["quick", "coder", "moe", "thinking"] as const).map((tier) => (
                <div key={tier} className="flex items-center gap-2">
                  <span className="w-16 text-sm text-muted-foreground capitalize">{tier}:</span>
                  {discoveredModels.length > 0 ? (
                    <select
                      value={formData.models[tier]}
                      onChange={(e) => setFormData(prev => ({
                        ...prev,
                        models: { ...prev.models, [tier]: e.target.value }
                      }))}
                      className="flex-1 px-2 py-1 text-sm border rounded bg-background"
                    >
                      <option value="">Select a model...</option>
                      {discoveredModels.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.name}
                          {model.sizeGB ? ` (${model.sizeGB})` : ""}
                          {model.description && !model.sizeGB ? ` - ${model.description}` : ""}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={formData.models[tier]}
                      onChange={(e) => setFormData(prev => ({
                        ...prev,
                        models: { ...prev.models, [tier]: e.target.value }
                      }))}
                      className="flex-1 px-2 py-1 text-sm border rounded bg-background"
                      placeholder={isGemini ? "gemini-2.0-flash" : "model name"}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          
          <div className="flex items-center justify-between pt-4 border-t">
            {backend && onDelete ? (
              <button
                type="button"
                onClick={() => onDelete(backend.id)}
                className="px-3 py-2 text-sm text-accent hover:bg-accent/10 rounded transition-colors"
              >
                🗑️ Remove
              </button>
            ) : (
              <div />
            )}
            <div className="flex gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm border rounded hover:bg-muted transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center gap-1"
              >
                <WatermelonSeed className="w-2 h-3" />
                {saving ? "Saving..." : backend ? "Save" : "Add"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
