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
import { readFile } from "fs/promises"
import { join } from "path"

// Paths derived from DELIA_DATA_DIR env var (default: ../data/)
const DATA_DIR = process.env.DELIA_DATA_DIR || join(process.cwd(), "..", "data")
const CACHE_DIR = join(DATA_DIR, "cache")
const STATS_FILE = join(CACHE_DIR, "usage_stats.json")
const ENHANCED_STATS_FILE = join(CACHE_DIR, "enhanced_stats.json")

interface ModelStats {
  calls: number
  tokens: number
}

interface UsageStats {
  quick: ModelStats
  coder: ModelStats
  moe: ModelStats
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
  }
}

export async function GET() {
  try {
    // Read basic stats
    const data = await readFile(STATS_FILE, "utf-8")
    const stats: UsageStats = JSON.parse(data)
    
    // Ensure all tiers exist with defaults (backward compatibility)
    if (!stats.quick) stats.quick = { calls: 0, tokens: 0 }
    if (!stats.coder) stats.coder = { calls: 0, tokens: 0 }
    if (!stats.moe) stats.moe = { calls: 0, tokens: 0 }
    
    // Try to read enhanced stats
    let enhanced: EnhancedStats | null = null
    try {
      const enhancedData = await readFile(ENHANCED_STATS_FILE, "utf-8")
      enhanced = JSON.parse(enhancedData)
    } catch {
      // Enhanced stats file may not exist yet
    }
    
    return NextResponse.json({
      stats,
      enhanced,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    // Return empty stats if file doesn't exist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        stats: {
          quick: { calls: 0, tokens: 0 },
          coder: { calls: 0, tokens: 0 },
          moe: { calls: 0, tokens: 0 },
        },
        enhanced: null,
        timestamp: new Date().toISOString(),
      })
    }
    
    return NextResponse.json(
      { error: "Failed to read stats file" },
      { status: 500 }
    )
  }
}
