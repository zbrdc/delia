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
import { getCacheDir, getLogsFile } from "@/lib/paths"

// Use same path resolution as CLI
const CACHE_DIR = getCacheDir()

interface LiveLog {
  ts: string
  type: string // INFO, MODEL, THINK, RESPONSE, QUEUE
  message: string
  model?: string
  tokens?: number
  provider?: string
  backend_id?: string
  backend?: string
  garden_msg?: string // Watermelon-themed fun message!
}

// Get logs file path for a specific provider/backend
function getLogsPath(provider?: string, backendId?: string): string {
  if (backendId) {
    return join(CACHE_DIR, `logs_${backendId}.json`)
  }
  if (provider) {
    return join(CACHE_DIR, `logs_${provider}.json`)
  }
  return join(CACHE_DIR, "live_logs.json")
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const provider = searchParams.get("provider") || undefined
    const backendId = searchParams.get("backend_id") || undefined
    const all = searchParams.get("all") === "true"

    // If requesting all logs, try to aggregate from multiple sources
    if (all) {
      const allLogs: LiveLog[] = []

      // Try main logs file
      try {
        const mainData = await readFile(join(CACHE_DIR, "live_logs.json"), "utf-8")
        const mainLogs: LiveLog[] = JSON.parse(mainData)
        allLogs.push(...mainLogs)
      } catch {
        // Ignore if file doesn't exist
      }

      // Sort by timestamp descending and limit
      allLogs.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime())

      return NextResponse.json({
        logs: allLogs.slice(0, 100),
        timestamp: new Date().toISOString(),
      })
    }

    // Get logs for specific provider/backend
    const logsPath = getLogsPath(provider, backendId)
    const data = await readFile(logsPath, "utf-8")
    const logs: LiveLog[] = JSON.parse(data)

    return NextResponse.json({
      logs,
      provider,
      backend_id: backendId,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    // Return empty logs if file doesn't exist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        logs: [],
        timestamp: new Date().toISOString(),
      })
    }

    return NextResponse.json(
      { error: "Failed to read logs file", logs: [] },
      { status: 500 }
    )
  }
}
