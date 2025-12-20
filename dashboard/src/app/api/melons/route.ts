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
import { getMelonsFile } from "@/lib/paths"

interface MelonStats {
  model_id: string
  task_type: string
  melons: number
  golden_melons: number
  total_responses: number
  successful_responses: number
}

interface MelonsData {
  stats: Record<string, MelonStats>
}

interface LeaderboardEntry {
  model_id: string
  task_type: string
  melons: number
  golden_melons: number
  total_melon_value: number
  success_rate: number
  total_responses: number
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const taskType = searchParams.get("task_type")
    const limit = parseInt(searchParams.get("limit") || "5", 10)

    const melonsFile = getMelonsFile()
    const data = await readFile(melonsFile, "utf-8")
    const melonsData: MelonsData = JSON.parse(data)

    // Convert stats to leaderboard entries
    let entries: LeaderboardEntry[] = Object.values(melonsData.stats).map((stat) => ({
      model_id: stat.model_id,
      task_type: stat.task_type,
      melons: stat.melons,
      golden_melons: stat.golden_melons,
      total_melon_value: stat.melons + (stat.golden_melons * 500),
      success_rate: stat.total_responses > 0 
        ? stat.successful_responses / stat.total_responses 
        : 0,
      total_responses: stat.total_responses,
    }))

    // Filter by task type if specified
    if (taskType) {
      entries = entries.filter((e) => e.task_type === taskType)
    }

    // Sort by total melon value (golden melons first, then regular melons)
    entries.sort((a, b) => {
      if (b.golden_melons !== a.golden_melons) {
        return b.golden_melons - a.golden_melons
      }
      if (b.melons !== a.melons) {
        return b.melons - a.melons
      }
      return b.success_rate - a.success_rate
    })

    // Limit results
    const leaderboard = entries.slice(0, limit)

    // Calculate totals
    const totalMelons = entries.reduce((sum, e) => sum + e.melons, 0)
    const totalGoldenMelons = entries.reduce((sum, e) => sum + e.golden_melons, 0)
    const totalValue = entries.reduce((sum, e) => sum + e.total_melon_value, 0)

    return NextResponse.json({
      leaderboard,
      totals: {
        melons: totalMelons,
        golden_melons: totalGoldenMelons,
        total_value: totalValue,
        models: new Set(entries.map(e => e.model_id)).size,
      },
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    // Return empty leaderboard if file doesn't exist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        leaderboard: [],
        totals: {
          melons: 0,
          golden_melons: 0,
          total_value: 0,
          models: 0,
        },
        timestamp: new Date().toISOString(),
      })
    }

    return NextResponse.json(
      { error: "Failed to read melons file" },
      { status: 500 }
    )
  }
}
