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

import { NextResponse } from "next/server"
import { readFile, writeFile, copyFile, access } from "fs/promises"
import { join } from "path"
import { getSettingsFile } from "@/lib/paths"

// Path to settings.json (uses same resolution as CLI)
const PROJECT_ROOT = join(process.cwd(), "..")
const SETTINGS_FILE = getSettingsFile()
const SETTINGS_EXAMPLE = join(PROJECT_ROOT, "settings.json.example")

interface ModelSettings {
  quick: string
  coder: string
  moe: string
  thinking: string
}

interface OllamaModel {
  name: string
  size: number
  modified_at: string
}

interface BackendConfig {
  id: string
  name: string
  provider: string
  type: string
  url: string
  enabled: boolean
  priority: number
  models: ModelSettings
}

interface SettingsConfig {
  backends: BackendConfig[]
  routing: {
    prefer_local: boolean
  }
}

async function loadSettings(): Promise<SettingsConfig> {
  try {
    const content = await readFile(SETTINGS_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    // settings.json doesn't exist - try to copy from example
    try {
      await access(SETTINGS_EXAMPLE)
      await copyFile(SETTINGS_EXAMPLE, SETTINGS_FILE)
      const content = await readFile(SETTINGS_FILE, "utf-8")
      return JSON.parse(content)
    } catch {
      // No example file either - return default
      return { backends: [], routing: { prefer_local: true } }
    }
  }
}

async function saveSettings(config: SettingsConfig): Promise<void> {
  await writeFile(SETTINGS_FILE, JSON.stringify(config, null, 2), "utf-8")
}

// Get active backend from settings
function getActiveBackend(config: SettingsConfig): BackendConfig | undefined {
  const enabled = config.backends.filter(b => b.enabled)
  if (enabled.length === 0) return undefined
  
  // Sort by priority (assuming 0 is highest, or just take first)
  // Matching python logic: priority sort
  enabled.sort((a, b) => (a.priority || 0) - (b.priority || 0))
  return enabled[0]
}

// Fetch available models from backend
async function fetchModels(backend: BackendConfig): Promise<OllamaModel[]> {
  if (backend.provider !== "ollama") return []
  
  try {
    const res = await fetch(`${backend.url}/api/tags`, {
      signal: AbortSignal.timeout(5000)
    })
    if (res.ok) {
      const data = await res.json()
      return data.models || []
    }
  } catch {
    // Backend not available
  }
  return []
}

export async function GET() {
  try {
    const config = await loadSettings()
    const activeBackend = getActiveBackend(config)
    
    if (!activeBackend) {
       return NextResponse.json({
        success: false,
        error: "No active backend found",
        models: { quick: "", coder: "", moe: "", thinking: "" },
        availableModels: [],
        ollamaConnected: false
      })
    }

    const models = activeBackend.models
    
    // Fetch available models from the active backend
    const backendModels = await fetchModels(activeBackend)
    const availableModels = backendModels.map(m => ({
      name: m.name,
      size: m.size,
      sizeGB: (m.size / 1024 / 1024 / 1024).toFixed(1)
    }))
    
    return NextResponse.json({
      success: true,
      models,
      availableModels,
      ollamaConnected: backendModels.length > 0
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: (error as Error).message,
    }, { status: 500 })
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { tier, model } = body as { tier: "quick" | "coder" | "moe" | "thinking", model: string }
    
    if (!tier || !model) {
      return NextResponse.json({
        success: false,
        error: "Missing tier or model parameter",
      }, { status: 400 })
    }
    
    if (!["quick", "coder", "moe", "thinking"].includes(tier)) {
      return NextResponse.json({
        success: false,
        error: "Invalid tier. Must be quick, coder, moe, or thinking",
      }, { status: 400 })
    }
    
    // Read current config
    const config = await loadSettings()
    
    // Find active backend to update
    // Note: We update the specific backend object in the array
    const enabled = config.backends.filter(b => b.enabled)
    if (enabled.length === 0) {
        return NextResponse.json({
            success: false,
            error: "No active backend to update",
        }, { status: 404 })
    }
    
    // Sort to find the active one
    enabled.sort((a, b) => (a.priority || 0) - (b.priority || 0))
    const activeId = enabled[0].id
    
    const backendIndex = config.backends.findIndex(b => b.id === activeId)
    if (backendIndex === -1) {
         return NextResponse.json({
            success: false,
            error: "Active backend not found in list",
        }, { status: 404 })
    }

    // Update the model
    if (!config.backends[backendIndex].models) {
        config.backends[backendIndex].models = { quick: "", coder: "", moe: "", thinking: "" }
    }
    config.backends[backendIndex].models[tier] = model
    
    // Write back
    await saveSettings(config)
    
    return NextResponse.json({
      success: true,
      message: `Updated ${tier} model to ${model} for backend ${config.backends[backendIndex].name}`,
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: (error as Error).message,
    }, { status: 500 })
  }
}
