/**
 * Copyright (C) 2024 Delia Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

"use client"

import { useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, BarChart, Bar, Cell } from "recharts"

// ============================================================================
// INTERFACES
// ============================================================================

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

interface RoutingIntelligence {
  hedging: HedgingConfig
  prewarm: PrewarmConfig
  affinity: {
    alpha: number
    trackedPairs: number
  }
  timestamp: string
}

// Playbooks
interface PlaybookSummary {
  name: string
  bullet_count: number
  path: string
}

interface PlaybookBullet {
  id: string
  content: string
  section?: string
  helpful_count: number
  harmful_count: number
  created_at: string
  last_used?: string
  utility_score: number
}

// Memories
interface MemorySummary {
  name: string
  size: number
  path: string
  modified: string
}

// ACE Metrics
interface ACEBulletInfo {
  id: string
  task_type: string
  utility: number
  helpful: number
  harmful: number
}

interface ACEPlaybookStats {
  total_bullets: number
  task_types: string[]
  task_type_counts: Record<string, number>
  top_bullets: ACEBulletInfo[]
  low_utility_count: number
}

interface ACEEffectivenessMetrics {
  total_helpful: number
  total_harmful: number
  total_feedback: number
  overall_score: number
}

interface ACEMetrics {
  playbook: ACEPlaybookStats
  effectiveness: ACEEffectivenessMetrics
  reflections: { recent: unknown[]; count: number }
  curator: { deduplication_enabled: boolean; similarity_threshold: number }
  project_path: string
}

// ============================================================================
// CONSTANTS
// ============================================================================

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

const TIER_COLORS = {
  quick: "#8BB5A6",
  coder: "#689B8A",
  moe: "#FF6B7A",
  thinking: "#E85566",
}

// ============================================================================
// COMPONENTS
// ============================================================================

function WatermelonSeed({ className = "", style = {} }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 10 16" className={className} style={style} fill="currentColor">
      <ellipse cx="5" cy="8" rx="4" ry="7" />
    </svg>
  )
}

type TabType = "overview" | "backends" | "ace"

// ============================================================================
// MAIN DASHBOARD
// ============================================================================

export default function Dashboard() {
  // Core state
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">("dark")
  const [activeTab, setActiveTab] = useState<TabType>("overview")
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [secondsAgo, setSecondsAgo] = useState(0)

  // Backend state
  const [backendsResponse, setBackendsResponse] = useState<BackendsResponse | null>(null)
  const [activeBackend, setActiveBackend] = useState<string>("")
  const [circuitBreaker, setCircuitBreaker] = useState<CircuitBreakerState | null>(null)
  const [routingIntelligence, setRoutingIntelligence] = useState<RoutingIntelligence | null>(null)
  const [editingBackend, setEditingBackend] = useState<BackendStatus | null>(null)
  const [showAddBackend, setShowAddBackend] = useState(false)

  // ACE state
  const [playbooks, setPlaybooks] = useState<PlaybookSummary[]>([])
  const [selectedPlaybook, setSelectedPlaybook] = useState<string | null>(null)
  const [playbookBullets, setPlaybookBullets] = useState<PlaybookBullet[]>([])
  const [memories, setMemories] = useState<MemorySummary[]>([])
  const [selectedMemory, setSelectedMemory] = useState<string | null>(null)
  const [memoryContent, setMemoryContent] = useState<string>("")
  const [editingMemory, setEditingMemory] = useState<boolean>(false)
  const [aceMetrics, setAceMetrics] = useState<ACEMetrics | null>(null)
  const [aceSubTab, setAceSubTab] = useState<"metrics" | "playbooks" | "memories">("metrics")

  // Timer for "seconds ago" display
  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000))
    }, 1000)
    return () => clearInterval(timer)
  }, [lastUpdated])

  // Theme handling
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

  // ============================================================================
  // FETCH FUNCTIONS
  // ============================================================================

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

  const fetchPlaybooks = useCallback(async () => {
    try {
      const res = await fetch("/api/playbooks")
      if (res.ok) {
        const data = await res.json()
        if (data.success) setPlaybooks(data.playbooks || [])
      }
    } catch { /* ignore */ }
  }, [])

  const fetchPlaybookDetail = useCallback(async (name: string) => {
    try {
      const res = await fetch(`/api/playbooks?name=${name}`)
      if (res.ok) {
        const data = await res.json()
        if (data.success) {
          setSelectedPlaybook(name)
          setPlaybookBullets(data.bullets || [])
        }
      }
    } catch { /* ignore */ }
  }, [])

  const fetchMemories = useCallback(async () => {
    try {
      const res = await fetch("/api/memories")
      if (res.ok) {
        const data = await res.json()
        if (data.success) setMemories(data.memories || [])
      }
    } catch { /* ignore */ }
  }, [])

  const fetchMemoryDetail = useCallback(async (name: string) => {
    try {
      const res = await fetch(`/api/memories?name=${name}`)
      if (res.ok) {
        const data = await res.json()
        if (data.success) {
          setSelectedMemory(name)
          setMemoryContent(data.content || "")
        }
      }
    } catch { /* ignore */ }
  }, [])

  const saveMemory = useCallback(async () => {
    if (!selectedMemory) return
    try {
      const res = await fetch("/api/memories", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selectedMemory, content: memoryContent })
      })
      if (res.ok) {
        setEditingMemory(false)
        fetchMemories()
      }
    } catch { /* ignore */ }
  }, [selectedMemory, memoryContent, fetchMemories])

  const fetchAceMetrics = useCallback(async () => {
    try {
      const res = await fetch("/api/ace/metrics")
      if (res.ok) {
        const data = await res.json()
        setAceMetrics(data)
      }
    } catch { /* ignore */ }
  }, [])

  // SSE for backends
  useEffect(() => {
    const eventSource = new EventSource("/api/sse/backends")
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as BackendsResponse
        if (data.backends) {
          setBackendsResponse(data)
          setActiveBackend(data.activeBackend || "")
          setLastUpdated(new Date())
          setLoading(false)
        }
      } catch { /* ignore */ }
    }
    eventSource.onerror = () => console.debug("SSE reconnecting...")

    // Initial load timeout
    setTimeout(() => setLoading(false), 2000)

    return () => eventSource.close()
  }, [])

  // Periodic fetches
  useEffect(() => {
    fetchCircuitBreaker()
    fetchRoutingIntelligence()
    const interval = setInterval(() => {
      fetchCircuitBreaker()
      fetchRoutingIntelligence()
    }, 10000)
    return () => clearInterval(interval)
  }, [fetchCircuitBreaker, fetchRoutingIntelligence])

  // Tab-specific fetches
  useEffect(() => {
    if (activeTab === "ace") {
      fetchPlaybooks()
      fetchMemories()
      fetchAceMetrics()
    }
  }, [activeTab, fetchPlaybooks, fetchMemories, fetchAceMetrics])

  // ============================================================================
  // RENDER HELPERS
  // ============================================================================

  const hasOpenCircuit = circuitBreaker && Object.entries(circuitBreaker).some(([key, status]) =>
    key !== 'active_backend' && key !== 'timestamp' && status?.circuit_open
  )

  const utilityColor = (utility: number) => {
    if (utility >= 0.7) return "#10b981"
    if (utility >= 0.4) return "#f59e0b"
    return "#ef4444"
  }

  // ============================================================================
  // LOADING / ERROR STATES
  // ============================================================================

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
            <button onClick={() => window.location.reload()} className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm">
              Retry
            </button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
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
                <span className="font-semibold">{backendsResponse?.summary?.available || 0}</span>
                <span className="text-muted-foreground">backends online</span>
              </span>
              {aceMetrics && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <span className="flex items-center gap-1">
                    <span className="font-semibold">{aceMetrics.playbook.total_bullets}</span>
                    <span className="text-muted-foreground">bullets</span>
                  </span>
                </>
              )}
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
              { id: "overview", label: "Overview", icon: "📊" },
              { id: "backends", label: "Backends", icon: "🌱" },
              { id: "ace", label: "ACE Framework", icon: "🎯" },
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

        {/* ================================================================== */}
        {/* OVERVIEW TAB */}
        {/* ================================================================== */}
        {activeTab === "overview" && (
          <div className="space-y-4">
            {/* Health Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Backends</CardDescription>
                  <CardTitle className="text-3xl">
                    {backendsResponse?.summary?.available || 0}/{backendsResponse?.summary?.enabled || 0}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    {backendsResponse?.summary?.local || 0} local · {backendsResponse?.summary?.remote || 0} remote
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>ACE Bullets</CardDescription>
                  <CardTitle className="text-3xl">{aceMetrics?.playbook.total_bullets || 0}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    {aceMetrics?.playbook.task_types.length || 0} task types
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Effectiveness</CardDescription>
                  <CardTitle className="text-3xl">
                    {aceMetrics ? `${(aceMetrics.effectiveness.overall_score * 100).toFixed(0)}%` : "—"}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    {aceMetrics?.effectiveness.total_helpful || 0} helpful · {aceMetrics?.effectiveness.total_harmful || 0} harmful
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Memories</CardDescription>
                  <CardTitle className="text-3xl">{memories.length}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    Persistent knowledge files
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Backend Status Grid */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <WatermelonSeed className="w-3 h-4" />
                  Backend Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {backendsResponse?.backends?.map((backend) => (
                    <div
                      key={backend.id}
                      className={`p-3 rounded-lg border ${
                        backend.health.available
                          ? activeBackend === backend.id ? "border-primary/50 bg-primary/5" : "border-border"
                          : "border-border/50 opacity-60"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${backend.health.available && activeBackend === backend.id ? 'animate-pulse' : ''}`}
                          style={{ backgroundColor: backend.health.available ? (PROVIDER_COLORS[backend.provider] || '#689B8A') : '#666' }}
                        />
                        <span className="font-medium text-sm flex-1">{backend.name}</span>
                        <Badge variant="outline" className="text-[10px]" style={{ borderColor: TYPE_COLORS[backend.type], color: TYPE_COLORS[backend.type] }}>
                          {backend.type}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                        <span>{backend.provider}</span>
                        {backend.health.response_time_ms > 0 && (
                          <>
                            <span>·</span>
                            <span>{backend.health.response_time_ms}ms</span>
                          </>
                        )}
                        {backend.health.circuit_open && <span className="text-destructive ml-1">⚠ circuit open</span>}
                      </div>
                    </div>
                  ))}
                  {!backendsResponse?.backends?.length && (
                    <div className="col-span-full h-24 flex items-center justify-center text-muted-foreground text-sm">
                      No backends configured
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* ACE Quick Stats */}
            {aceMetrics && aceMetrics.playbook.top_bullets.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Top Performing Bullets</CardTitle>
                  <CardDescription className="text-xs">Highest utility based on feedback</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {aceMetrics.playbook.top_bullets.slice(0, 5).map((bullet) => (
                      <div key={bullet.id} className="flex items-center justify-between p-2 rounded border">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">{bullet.task_type}</Badge>
                          <span className="text-xs font-mono text-muted-foreground">{bullet.id}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-xs">
                            <span className="text-green-600">↑{bullet.helpful}</span>
                            {" / "}
                            <span className="text-red-600">↓{bullet.harmful}</span>
                          </span>
                          <Badge style={{ backgroundColor: utilityColor(bullet.utility) }} className="text-white text-xs">
                            {(bullet.utility * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* BACKENDS TAB */}
        {/* ================================================================== */}
        {activeTab === "backends" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Backends List */}
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <WatermelonSeed className="w-3 h-4" />
                    Backends
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

                {/* Model Tier Config */}
                <div className="pt-2 border-t">
                  <h4 className="text-sm font-medium mb-3">Model Tiers</h4>
                  <div className="space-y-2">
                    {(["quick", "coder", "moe", "thinking"] as const).map((tier) => {
                      const backend = backendsResponse?.backends?.find(b => b.id === activeBackend)
                      const model = backend?.models?.[tier] || "—"
                      return (
                        <div key={tier} className="flex items-center justify-between text-sm">
                          <span className="capitalize" style={{ color: TIER_COLORS[tier] }}>{tier}</span>
                          <span className="text-muted-foreground text-xs truncate max-w-[200px]">{model}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ================================================================== */}
        {/* ACE FRAMEWORK TAB */}
        {/* ================================================================== */}
        {activeTab === "ace" && (
          <div className="space-y-4">
            {/* Sub-tab navigation */}
            <div className="flex gap-2">
              <button
                onClick={() => setAceSubTab("metrics")}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  aceSubTab === "metrics" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                }`}
              >
                📈 Metrics
              </button>
              <button
                onClick={() => setAceSubTab("playbooks")}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  aceSubTab === "playbooks" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                }`}
              >
                📚 Playbooks ({playbooks.length})
              </button>
              <button
                onClick={() => setAceSubTab("memories")}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  aceSubTab === "memories" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                }`}
              >
                🧠 Memories ({memories.length})
              </button>
            </div>

            {/* Metrics Sub-tab */}
            {aceSubTab === "metrics" && (
              <div className="space-y-4">
                {/* Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardDescription>Total Bullets</CardDescription>
                      <CardTitle className="text-3xl">{aceMetrics?.playbook.total_bullets || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground">
                        Across {aceMetrics?.playbook.task_types.length || 0} task types
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardDescription>Effectiveness Score</CardDescription>
                      <CardTitle className="text-3xl">
                        {aceMetrics ? `${(aceMetrics.effectiveness.overall_score * 100).toFixed(0)}%` : "—"}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground">
                        {aceMetrics?.effectiveness.total_helpful || 0} helpful / {aceMetrics?.effectiveness.total_harmful || 0} harmful
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardDescription>Low Utility Bullets</CardDescription>
                      <CardTitle className="text-3xl">{aceMetrics?.playbook.low_utility_count || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground">
                        Candidates for pruning (&lt; 0.3)
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardDescription>Recent Reflections</CardDescription>
                      <CardTitle className="text-3xl">{aceMetrics?.reflections.count || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xs text-muted-foreground">
                        Learning loop activity
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Bullets by Task Type Chart */}
                {aceMetrics && Object.keys(aceMetrics.playbook.task_type_counts).length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Bullets by Task Type</CardTitle>
                      <CardDescription className="text-xs">
                        Distribution of playbook bullets across categories
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart
                          data={Object.entries(aceMetrics.playbook.task_type_counts).map(([name, count]) => ({ name, count }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            dataKey="name"
                            className="text-xs"
                            angle={-45}
                            textAnchor="end"
                            height={80}
                          />
                          <YAxis className="text-xs" />
                          <Tooltip />
                          <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                )}

                {/* Top Performing Bullets */}
                {aceMetrics && aceMetrics.playbook.top_bullets.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Top Performing Bullets</CardTitle>
                      <CardDescription className="text-xs">
                        Highest utility bullets based on helpful/harmful feedback
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {aceMetrics.playbook.top_bullets.slice(0, 10).map((bullet) => (
                          <div
                            key={bullet.id}
                            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                          >
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">{bullet.task_type}</Badge>
                              <span className="text-xs font-mono text-muted-foreground">{bullet.id}</span>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className="text-sm">
                                <span className="text-green-600">↑{bullet.helpful}</span>
                                {" / "}
                                <span className="text-red-600">↓{bullet.harmful}</span>
                              </span>
                              <Badge style={{ backgroundColor: utilityColor(bullet.utility) }} className="text-white">
                                {(bullet.utility * 100).toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Curator Config */}
                {aceMetrics && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Curator Configuration</CardTitle>
                      <CardDescription className="text-xs">Semantic deduplication settings</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Deduplication</p>
                          <p className="text-lg font-medium">
                            {aceMetrics.curator.deduplication_enabled ? "Enabled" : "Disabled"}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Similarity Threshold</p>
                          <p className="text-lg font-medium">
                            {(aceMetrics.curator.similarity_threshold * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {/* Playbooks Sub-tab */}
            {aceSubTab === "playbooks" && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Playbook List */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Playbooks</CardTitle>
                    <CardDescription className="text-xs">Strategic bullets by task type</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {playbooks.map((pb) => (
                        <button
                          key={pb.name}
                          onClick={() => fetchPlaybookDetail(pb.name)}
                          className={`w-full text-left p-2 rounded text-sm transition-colors ${
                            selectedPlaybook === pb.name ? "bg-primary/10 text-primary" : "hover:bg-muted"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{pb.name}</span>
                            <Badge variant="secondary" className="text-xs">{pb.bullet_count}</Badge>
                          </div>
                        </button>
                      ))}
                      {!playbooks.length && (
                        <div className="h-24 flex items-center justify-center text-muted-foreground text-sm">
                          No playbooks found
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Bullets Detail */}
                <Card className="lg:col-span-2">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">
                      {selectedPlaybook ? `${selectedPlaybook} Bullets` : "Select a Playbook"}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {selectedPlaybook && playbookBullets.length > 0 ? (
                      <div className="space-y-2 max-h-[500px] overflow-y-auto">
                        {playbookBullets.map((bullet) => (
                          <div key={bullet.id} className="p-3 rounded border text-sm">
                            <p>{bullet.content}</p>
                            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                              <span className="font-mono">{bullet.id}</span>
                              <span>·</span>
                              <span className="text-green-600">↑{bullet.helpful_count}</span>
                              <span className="text-red-600">↓{bullet.harmful_count}</span>
                              <span>·</span>
                              <span>utility: {(bullet.utility_score * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
                        {selectedPlaybook ? "No bullets in this playbook" : "Select a playbook to view bullets"}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Memories Sub-tab */}
            {aceSubTab === "memories" && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Memory List */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Memories</CardTitle>
                    <CardDescription className="text-xs">Persistent project knowledge</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {memories.map((mem) => (
                        <button
                          key={mem.name}
                          onClick={() => fetchMemoryDetail(mem.name)}
                          className={`w-full text-left p-2 rounded text-sm transition-colors ${
                            selectedMemory === mem.name ? "bg-primary/10 text-primary" : "hover:bg-muted"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{mem.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {(mem.size / 1024).toFixed(1)}kb
                            </span>
                          </div>
                        </button>
                      ))}
                      {!memories.length && (
                        <div className="h-24 flex items-center justify-center text-muted-foreground text-sm">
                          No memories found
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Memory Content */}
                <Card className="lg:col-span-2">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">
                        {selectedMemory ? `${selectedMemory}.md` : "Select a Memory"}
                      </CardTitle>
                      {selectedMemory && (
                        <div className="flex gap-2">
                          {editingMemory ? (
                            <>
                              <button
                                onClick={saveMemory}
                                className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground"
                              >
                                Save
                              </button>
                              <button
                                onClick={() => setEditingMemory(false)}
                                className="text-xs px-2 py-1 rounded bg-muted"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => setEditingMemory(true)}
                              className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80"
                            >
                              Edit
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    {selectedMemory ? (
                      editingMemory ? (
                        <textarea
                          value={memoryContent}
                          onChange={(e) => setMemoryContent(e.target.value)}
                          className="w-full h-[400px] p-3 rounded border bg-background font-mono text-sm resize-none"
                        />
                      ) : (
                        <pre className="whitespace-pre-wrap text-sm max-h-[400px] overflow-y-auto p-3 rounded bg-muted/30">
                          {memoryContent || "Empty memory"}
                        </pre>
                      )
                    ) : (
                      <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
                        Select a memory to view content
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t py-2">
        <div className="container mx-auto px-4 text-center text-xs text-muted-foreground">
          {aceMetrics?.project_path && <span>Project: {aceMetrics.project_path.split('/').pop()}</span>}
        </div>
      </footer>

      {/* Add Backend Modal Placeholder */}
      {showAddBackend && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowAddBackend(false)}>
          <Card className="w-full max-w-md" onClick={e => e.stopPropagation()}>
            <CardHeader>
              <CardTitle>Add Backend</CardTitle>
              <CardDescription>Configure a new LLM backend</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Configure backends in ~/.delia/settings.json
              </p>
              <button
                onClick={() => setShowAddBackend(false)}
                className="mt-4 w-full py-2 rounded bg-muted text-sm"
              >
                Close
              </button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Edit Backend Modal Placeholder */}
      {editingBackend && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setEditingBackend(null)}>
          <Card className="w-full max-w-md" onClick={e => e.stopPropagation()}>
            <CardHeader>
              <CardTitle>{editingBackend.name}</CardTitle>
              <CardDescription>{editingBackend.provider} · {editingBackend.type}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground">URL</p>
                <p className="text-sm font-mono">{editingBackend.url}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Status</p>
                <p className="text-sm">{editingBackend.health.available ? "Online" : "Offline"}</p>
              </div>
              {editingBackend.health.loaded_models?.length > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Loaded Models</p>
                  <div className="flex flex-wrap gap-1">
                    {editingBackend.health.loaded_models.map((m, i) => (
                      <Badge key={i} variant="secondary" className="text-xs">{m}</Badge>
                    ))}
                  </div>
                </div>
              )}
              <button
                onClick={() => setEditingBackend(null)}
                className="mt-4 w-full py-2 rounded bg-muted text-sm"
              >
                Close
              </button>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
