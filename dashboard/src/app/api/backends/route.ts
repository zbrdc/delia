import { NextResponse } from "next/server"
import { readFile, writeFile } from "fs/promises"
import { join } from "path"

// Path to settings.json (always in project root)
const PROJECT_ROOT = join(process.cwd(), "..")
const SETTINGS_FILE = join(PROJECT_ROOT, "settings.json")

interface BackendModels {
  quick: string
  coder: string
  moe: string
  thinking: string
}

interface BackendConfig {
  id: string
  name: string
  provider: "ollama" | "llamacpp" | "vllm" | "openai" | "gemini" | "custom"
  type: "local" | "remote"
  url: string
  enabled: boolean
  priority: number
  models: BackendModels
  health_endpoint?: string
  models_endpoint?: string
  context_limit?: number
  api_key?: string
}

interface BackendStatus extends BackendConfig {
  health: {
    available: boolean
    response_time_ms: number
    last_error: string
    loaded_models: string[]
    circuit_open: boolean
  }
}

// Full settings structure matches settings.json
interface SettingsConfig {
  version: string
  system?: any
  backends: BackendConfig[]
  routing: {
    prefer_local: boolean
    fallback_enabled: boolean
    load_balance: boolean
  }
  models?: any
  generation?: any
  costs?: any
  auth?: any
}

// Provider-specific health check endpoints
const PROVIDER_HEALTH_ENDPOINTS: Record<string, string> = {
  ollama: "/api/tags",
  llamacpp: "/health",
  vllm: "/health",
  openai: "",  // No health check for OpenAI
  gemini: "",  // Gemini uses API key validation instead
  custom: "/health",
}

// Provider-specific model list endpoints
const PROVIDER_MODEL_ENDPOINTS: Record<string, string> = {
  ollama: "/api/tags",
  llamacpp: "/v1/models",
  vllm: "/v1/models",
  openai: "/v1/models",
  gemini: "/v1beta/models",
  custom: "/v1/models",
}

async function loadSettingsConfig(): Promise<SettingsConfig> {
  try {
    const content = await readFile(SETTINGS_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    // Return default config if file doesn't exist
    return {
      version: "1.0",
      system: {
        gpu_memory_limit_gb: 8,
        memory_buffer_gb: 1,
        max_concurrent_requests_per_backend: 1,
      },
      backends: [],
      routing: {
        prefer_local: true,
        fallback_enabled: true,
        load_balance: false,
      },
      models: {},
      auth: {
        enabled: false,
        tracking_enabled: true
      }
    }
  }
}

async function saveSettingsConfig(config: SettingsConfig): Promise<void> {
  await writeFile(SETTINGS_FILE, JSON.stringify(config, null, 2), "utf-8")
}

async function checkBackendHealth(backend: BackendConfig): Promise<BackendStatus> {
  const healthEndpoint = backend.health_endpoint || PROVIDER_HEALTH_ENDPOINTS[backend.provider] || "/health"
  const startTime = Date.now()

  const status: BackendStatus = {
    ...backend,
    health: {
      available: false,
      response_time_ms: 0,
      last_error: "",
      loaded_models: [],
      circuit_open: false,
    }
  }

  if (!backend.enabled) {
    status.health.last_error = "Disabled"
    return status
  }

  // Special handling for Gemini - validate API key
  if (backend.provider === "gemini") {
    if (!backend.api_key) {
      status.health.last_error = "No API key"
      return status
    }
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${backend.api_key}`,
        { signal: controller.signal }
      )
      clearTimeout(timeoutId)

      status.health.response_time_ms = Date.now() - startTime

      if (response.ok) {
        status.health.available = true
        const data = await response.json()
        if (data.models) {
          status.health.loaded_models = data.models
            .filter((m: { supportedGenerationMethods?: string[] }) =>
              m.supportedGenerationMethods?.includes("generateContent"))
            .map((m: { name: string }) => m.name.replace("models/", ""))
            .slice(0, 10) // Limit to first 10
        }
      } else {
        status.health.last_error = response.status === 400 ? "Invalid API key" : `HTTP ${response.status}`
      }
    } catch (error) {
      status.health.response_time_ms = Date.now() - startTime
      status.health.last_error = error instanceof Error ? error.message : "Connection failed"
    }
    return status
  }

  // Skip health check for OpenAI (no health endpoint)
  if (!healthEndpoint) {
    status.health.available = true
    return status
  }

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)

    const response = await fetch(`${backend.url}${healthEndpoint}`, {
      signal: controller.signal
    })
    clearTimeout(timeoutId)

    status.health.response_time_ms = Date.now() - startTime

    if (response.ok) {
      status.health.available = true

      // Parse loaded models from response
      try {
        const data = await response.json()
        if (backend.provider === "ollama" && data.models) {
          status.health.loaded_models = data.models.map((m: { name: string }) => m.name)
        } else if (data.data) {
          // OpenAI-compatible format
          status.health.loaded_models = data.data.map((m: { id: string }) => m.id)
        }
      } catch {
        // Ignore JSON parsing errors
      }
    } else {
      status.health.last_error = `HTTP ${response.status}`
    }
  } catch (error) {
    status.health.response_time_ms = Date.now() - startTime
    status.health.last_error = error instanceof Error ? error.message : "Connection failed"
  }

  return status
}

export async function GET() {
  try {
    const config = await loadSettingsConfig()
    
    // Check all backends in parallel
    const statusPromises = config.backends.map(backend => checkBackendHealth(backend))
    const statuses = await Promise.all(statusPromises)
    
    // Count available backends
    const availableCount = statuses.filter(s => s.health.available).length
    const localBackends = statuses.filter(s => s.type === "local")
    const remoteBackends = statuses.filter(s => s.type === "remote")
    
    // Determine active backend (first available, prefer local if configured)
    let activeBackend = ""
    if (config.routing.prefer_local) {
      const availableLocal = localBackends.find(b => b.health.available)
      activeBackend = availableLocal?.id || statuses.find(s => s.health.available)?.id || ""
    } else {
      activeBackend = statuses.find(s => s.health.available)?.id || ""
    }
    
    return NextResponse.json({
      backends: statuses,
      routing: config.routing,
      activeBackend,
      summary: {
        total: config.backends.length,
        enabled: config.backends.filter(b => b.enabled).length,
        available: availableCount,
        local: localBackends.length,
        remote: remoteBackends.length,
      },
      timestamp: new Date().toISOString()
    }, {
      headers: {
        "Cache-Control": "max-age=5, stale-while-revalidate=10"
      }
    })
  } catch (error) {
    return NextResponse.json(
      { 
        error: "Failed to check backend status",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    )
  }
}

// POST: Add a new backend
export async function POST(request: Request) {
  try {
    const body = await request.json() as Partial<BackendConfig>
    
    // Validate required fields
    if (!body.id || !body.name || !body.provider || !body.type || !body.url) {
      return NextResponse.json({
        success: false,
        error: "Missing required fields: id, name, provider, type, url"
      }, { status: 400 })
    }
    
    // Validate provider
    if (!["ollama", "llamacpp", "vllm", "openai", "gemini", "custom"].includes(body.provider)) {
      return NextResponse.json({
        success: false,
        error: "Invalid provider. Must be: ollama, llamacpp, vllm, openai, gemini, or custom"
      }, { status: 400 })
    }
    
    // Validate type
    if (!["local", "remote"].includes(body.type)) {
      return NextResponse.json({
        success: false,
        error: "Invalid type. Must be: local or remote"
      }, { status: 400 })
    }
    
    const config = await loadSettingsConfig()
    
    // Check for duplicate ID
    if (config.backends.some(b => b.id === body.id)) {
      return NextResponse.json({
        success: false,
        error: `Backend with ID '${body.id}' already exists`
      }, { status: 409 })
    }
    
    // Create new backend with defaults
    const newBackend: BackendConfig = {
      id: body.id,
      name: body.name,
      provider: body.provider,
      type: body.type,
      url: body.url,
      enabled: body.enabled ?? true,
      priority: body.priority ?? config.backends.length,
      models: body.models || { quick: "", coder: "", moe: "", thinking: "" },
      health_endpoint: body.health_endpoint || PROVIDER_HEALTH_ENDPOINTS[body.provider],
      models_endpoint: body.models_endpoint || PROVIDER_MODEL_ENDPOINTS[body.provider],
      context_limit: body.context_limit || 32768,
      ...(body.api_key && { api_key: body.api_key }),
    }
    
    config.backends.push(newBackend)
    await saveSettingsConfig(config)
    
    return NextResponse.json({
      success: true,
      message: `Backend '${newBackend.name}' added`,
      backend: newBackend
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

// PUT: Update an existing backend
export async function PUT(request: Request) {
  try {
    const body = await request.json() as Partial<BackendConfig> & { id: string }
    
    if (!body.id) {
      return NextResponse.json({
        success: false,
        error: "Missing required field: id"
      }, { status: 400 })
    }
    
    const config = await loadSettingsConfig()
    const index = config.backends.findIndex(b => b.id === body.id)
    
    if (index === -1) {
      return NextResponse.json({
        success: false,
        error: `Backend with ID '${body.id}' not found`
      }, { status: 404 })
    }
    
    // Update fields (preserve existing values for undefined fields)
    const existing = config.backends[index]
    config.backends[index] = {
      ...existing,
      name: body.name ?? existing.name,
      provider: body.provider ?? existing.provider,
      type: body.type ?? existing.type,
      url: body.url ?? existing.url,
      enabled: body.enabled ?? existing.enabled,
      priority: body.priority ?? existing.priority,
      models: body.models ?? existing.models,
      health_endpoint: body.health_endpoint ?? existing.health_endpoint,
      models_endpoint: body.models_endpoint ?? existing.models_endpoint,
      context_limit: body.context_limit ?? existing.context_limit,
      api_key: body.api_key ?? existing.api_key,
    }
    
    await saveSettingsConfig(config)
    
    return NextResponse.json({
      success: true,
      message: `Backend '${body.id}' updated`,
      backend: config.backends[index]
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

// DELETE: Remove a backend
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get("id")
    
    if (!id) {
      return NextResponse.json({
        success: false,
        error: "Missing required query parameter: id"
      }, { status: 400 })
    }
    
    const config = await loadSettingsConfig()
    const index = config.backends.findIndex(b => b.id === id)
    
    if (index === -1) {
      return NextResponse.json({
        success: false,
        error: `Backend with ID '${id}' not found`
      }, { status: 404 })
    }
    
    const removed = config.backends.splice(index, 1)[0]
    await saveSettingsConfig(config)
    
    return NextResponse.json({
      success: true,
      message: `Backend '${removed.name}' removed`,
      backend: removed
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

// PATCH: Update routing settings
export async function PATCH(request: Request) {
  try {
    const body = await request.json() as Partial<SettingsConfig["routing"]>
    
    const config = await loadSettingsConfig()
    
    // Update routing settings
    if (typeof body.prefer_local === "boolean") {
      config.routing.prefer_local = body.prefer_local
    }
    if (typeof body.fallback_enabled === "boolean") {
      config.routing.fallback_enabled = body.fallback_enabled
    }
    if (typeof body.load_balance === "boolean") {
      config.routing.load_balance = body.load_balance
    }
    
    await saveSettingsConfig(config)
    
    return NextResponse.json({
      success: true,
      message: "Routing settings updated",
      routing: config.routing
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}
