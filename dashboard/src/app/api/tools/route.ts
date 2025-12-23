/**
 * Copyright (C) 2024 Delia Contributors
 * GPL-3.0-or-later
 */

import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { getEnhancedStatsFile } from "@/lib/paths"

interface ToolStats {
  calls: number
  errors: number
  avg_ms: number
  total_ms: number
}

interface RecentToolCall {
  ts: string
  tool: string
  ms: number
  success: boolean
  error?: string
}

export async function GET() {
  try {
    const statsFile = getEnhancedStatsFile()
    const data = await readFile(statsFile, "utf-8")
    const stats = JSON.parse(data)

    const toolStats: Record<string, ToolStats> = stats.tool_stats || {}
    const recentCalls: RecentToolCall[] = stats.recent_tool_calls || []

    // Sort tools by call count descending
    const sortedTools = Object.entries(toolStats)
      .sort((a, b) => b[1].calls - a[1].calls)
      .map(([name, stats]) => ({ name, ...stats }))

    // Calculate summary
    const totalCalls = Object.values(toolStats).reduce((sum, t) => sum + t.calls, 0)
    const totalErrors = Object.values(toolStats).reduce((sum, t) => sum + t.errors, 0)
    const uniqueTools = Object.keys(toolStats).length

    return NextResponse.json({
      success: true,
      tools: sortedTools,
      recent: recentCalls.slice(-50).reverse(),  // Last 50, newest first
      summary: {
        total_calls: totalCalls,
        total_errors: totalErrors,
        unique_tools: uniqueTools,
        error_rate: totalCalls > 0 ? (totalErrors / totalCalls * 100).toFixed(2) + "%" : "0%",
      },
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        success: true,
        tools: [],
        recent: [],
        summary: { total_calls: 0, total_errors: 0, unique_tools: 0, error_rate: "0%" },
      })
    }
    return NextResponse.json({ success: false, error: "Failed to read tool stats" }, { status: 500 })
  }
}
