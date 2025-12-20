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

import { readFile } from "fs/promises"
import { watch } from "fs"
import { getSettingsFile } from "@/lib/paths"

// Use same path resolution as CLI
const SETTINGS_FILE = getSettingsFile()

interface BackendConfig {
  id: string
  name: string
  provider: "ollama" | "llamacpp" | "vllm" | "openai" | "gemini" | "custom"
  type: "local" | "remote"
  url: string
  enabled: boolean
  priority: number
  models: { quick: string; coder: string; moe: string; thinking: string }
  health_endpoint?: string
  api_key?: string
}

interface SettingsConfig {
  backends: BackendConfig[]
  routing: {
    prefer_local: boolean
    fallback_enabled: boolean
    load_balance: boolean
  }
}

const PROVIDER_HEALTH_ENDPOINTS: Record<string, string> = {
  ollama: "/api/tags",
  llamacpp: "/health",
  vllm: "/health",
  openai: "",
  gemini: "",
  custom: "/health",
}

async function loadSettings(): Promise<SettingsConfig> {
  try {
    const content = await readFile(SETTINGS_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    return { backends: [], routing: { prefer_local: true, fallback_enabled: true, load_balance: false } }
  }
}

async function checkBackendHealth(backend: BackendConfig) {
  const healthEndpoint = backend.health_endpoint || PROVIDER_HEALTH_ENDPOINTS[backend.provider] || "/health"
  const startTime = Date.now()

  const health = {
    available: false,
    response_time_ms: 0,
    last_error: "",
    loaded_models: [] as string[],
    circuit_open: false,
  }

  if (!backend.enabled) {
    health.last_error = "Disabled"
    return { ...backend, health }
  }

  // Gemini - validate API key
  if (backend.provider === "gemini") {
    if (!backend.api_key) {
      health.last_error = "No API key"
      return { ...backend, health }
    }
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 3000)
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${backend.api_key}`,
        { signal: controller.signal }
      )
      clearTimeout(timeoutId)
      health.response_time_ms = Date.now() - startTime
      if (response.ok) {
        health.available = true
        const data = await response.json()
        if (data.models) {
          health.loaded_models = data.models
            .filter((m: { supportedGenerationMethods?: string[] }) =>
              m.supportedGenerationMethods?.includes("generateContent"))
            .map((m: { name: string }) => m.name.replace("models/", ""))
            .slice(0, 10)
        }
      } else {
        health.last_error = response.status === 400 ? "Invalid API key" : `HTTP ${response.status}`
      }
    } catch (error) {
      health.response_time_ms = Date.now() - startTime
      health.last_error = error instanceof Error ? error.message : "Connection failed"
    }
    return { ...backend, health }
  }

  // Skip health check for OpenAI
  if (!healthEndpoint) {
    health.available = true
    return { ...backend, health }
  }

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000)
    const response = await fetch(`${backend.url}${healthEndpoint}`, { signal: controller.signal })
    clearTimeout(timeoutId)
    health.response_time_ms = Date.now() - startTime

    if (response.ok) {
      health.available = true
      try {
        const data = await response.json()
        if (backend.provider === "ollama" && data.models) {
          health.loaded_models = data.models.map((m: { name: string }) => m.name)
        } else if (data.data) {
          health.loaded_models = data.data.map((m: { id: string }) => m.id)
        }
      } catch { /* ignore */ }
    } else {
      health.last_error = `HTTP ${response.status}`
    }
  } catch (error) {
    health.response_time_ms = Date.now() - startTime
    health.last_error = error instanceof Error ? error.message : "Connection failed"
  }

  return { ...backend, health }
}

async function getFullState() {
  const config = await loadSettings()
  const statuses = await Promise.all(config.backends.map(checkBackendHealth))

  const availableCount = statuses.filter(s => s.health.available).length
  const localBackends = statuses.filter(s => s.type === "local")

  let activeBackend = ""
  if (config.routing.prefer_local) {
    const availableLocal = localBackends.find(b => b.health.available)
    activeBackend = availableLocal?.id || statuses.find(s => s.health.available)?.id || ""
  } else {
    activeBackend = statuses.find(s => s.health.available)?.id || ""
  }

  return {
    backends: statuses,
    routing: config.routing,
    activeBackend,
    summary: {
      total: config.backends.length,
      enabled: config.backends.filter(b => b.enabled).length,
      available: availableCount,
      local: localBackends.length,
      remote: statuses.filter(s => s.type === "remote").length,
    },
    timestamp: new Date().toISOString()
  }
}

export async function GET() {
  const encoder = new TextEncoder()

  let isOpen = true
  let watcher: ReturnType<typeof watch> | null = null
  let healthInterval: NodeJS.Timeout | null = null

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: object) => {
        if (!isOpen) return
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`))
        } catch { /* client disconnected */ }
      }

      // Send initial state
      const initialState = await getFullState()
      send(initialState)

      // Watch settings.json for changes
      try {
        let debounceTimer: NodeJS.Timeout | null = null
        watcher = watch(SETTINGS_FILE, async (eventType) => {
          if (eventType === "change") {
            // Debounce rapid changes
            if (debounceTimer) clearTimeout(debounceTimer)
            debounceTimer = setTimeout(async () => {
              const state = await getFullState()
              send(state)
            }, 100)
          }
        })
      } catch {
        // File watch not available, fall back to polling
      }

      // Health check every 2s (backends may come online/offline)
      healthInterval = setInterval(async () => {
        if (!isOpen) return
        const state = await getFullState()
        send(state)
      }, 2000)
    },

    cancel() {
      isOpen = false
      if (watcher) watcher.close()
      if (healthInterval) clearInterval(healthInterval)
    }
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  })
}
