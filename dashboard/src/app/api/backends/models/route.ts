import { NextResponse } from "next/server"

// Provider-specific model list endpoints
const PROVIDER_MODEL_ENDPOINTS: Record<string, string> = {
  ollama: "/api/tags",
  llamacpp: "/v1/models",
  vllm: "/v1/models",
  openai: "/v1/models",
  custom: "/v1/models",
}

// Known Gemini models with their capabilities
const GEMINI_MODELS = [
  { name: "gemini-2.0-flash", description: "Fast, efficient model for most tasks" },
  { name: "gemini-2.0-flash-lite", description: "Lightweight version, faster responses" },
  { name: "gemini-1.5-flash", description: "Balanced speed and capability" },
  { name: "gemini-1.5-flash-8b", description: "Smaller, faster variant" },
  { name: "gemini-1.5-pro", description: "Most capable, for complex tasks" },
  { name: "gemini-1.0-pro", description: "Stable, reliable performance" },
]

interface ModelInfo {
  name: string
  size?: number
  sizeGB?: string
  modified_at?: string
  description?: string
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const url = searchParams.get("url")
    const provider = searchParams.get("provider") || "ollama"
    const apiKey = searchParams.get("apiKey")

    // Special handling for Gemini - return known models list
    // Gemini API requires authentication to list models, so we provide a curated list
    if (provider === "gemini") {
      // If API key provided, verify it works by making a test call
      if (apiKey) {
        try {
          const testUrl = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 5000)

          const response = await fetch(testUrl, { signal: controller.signal })
          clearTimeout(timeoutId)

          if (response.ok) {
            const data = await response.json()
            // Parse actual models from Gemini API
            const models: ModelInfo[] = data.models
              ?.filter((m: { name: string; supportedGenerationMethods?: string[] }) =>
                m.supportedGenerationMethods?.includes("generateContent")
              )
              .map((m: { name: string; displayName?: string; description?: string }) => ({
                name: m.name.replace("models/", ""),
                description: m.displayName || m.description
              })) || []

            return NextResponse.json({
              success: true,
              models: models.length > 0 ? models : GEMINI_MODELS,
              provider: "gemini",
              verified: true
            })
          }
        } catch {
          // Fall through to return default models
        }
      }

      // Return curated list if no API key or verification failed
      return NextResponse.json({
        success: true,
        models: GEMINI_MODELS,
        provider: "gemini",
        verified: false,
        note: "Using curated model list. Add API key to verify availability."
      })
    }

    if (!url) {
      return NextResponse.json({
        success: false,
        error: "Missing required parameter: url"
      }, { status: 400 })
    }

    const modelsEndpoint = PROVIDER_MODEL_ENDPOINTS[provider] || "/v1/models"
    const fullUrl = `${url}${modelsEndpoint}`

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)

    try {
      const response = await fetch(fullUrl, {
        signal: controller.signal
      })
      clearTimeout(timeoutId)

      if (!response.ok) {
        return NextResponse.json({
          success: false,
          error: `Backend returned ${response.status}`,
          models: []
        })
      }

      const data = await response.json()
      let models: ModelInfo[] = []

      // Parse models based on provider format
      if (provider === "ollama" && data.models) {
        models = data.models.map((m: { name: string; size?: number; modified_at?: string }) => ({
          name: m.name,
          size: m.size,
          sizeGB: m.size ? `${(m.size / 1024 / 1024 / 1024).toFixed(1)}GB` : undefined,
          modified_at: m.modified_at
        }))
      } else if (data.data) {
        // OpenAI-compatible format (llama.cpp, vLLM)
        models = data.data.map((m: { id: string; object?: string }) => ({
          name: m.id,
        }))
      } else if (Array.isArray(data)) {
        models = data.map((m: string | { name?: string; id?: string }) => ({
          name: typeof m === "string" ? m : (m.name || m.id || "unknown")
        }))
      }

      return NextResponse.json({
        success: true,
        models,
        provider,
        url
      })
    } catch (fetchError) {
      clearTimeout(timeoutId)
      return NextResponse.json({
        success: false,
        error: fetchError instanceof Error ? fetchError.message : "Connection failed",
        models: []
      })
    }
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      models: []
    }, { status: 500 })
  }
}
