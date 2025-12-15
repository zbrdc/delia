/**
 * Copyright (C) 2024 Delia Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

import { NextResponse } from "next/server"
import { readFile, writeFile } from "fs/promises"
import { join } from "path"
import { homedir } from "os"

// Paths - settings.json in project root, data in ~/.cache/delia
const PROJECT_ROOT = join(process.cwd(), "..")
const SETTINGS_FILE = join(PROJECT_ROOT, "settings.json")
const DATA_DIR = process.env.DELIA_DATA_DIR || join(homedir(), ".cache", "delia")
const CACHE_DIR = join(DATA_DIR, "cache")
const AFFINITY_FILE = join(CACHE_DIR, "affinity.json")
const PREWARM_FILE = join(CACHE_DIR, "prewarm.json")

interface AffinityData {
  alpha: number
  scores: Record<string, number>  // "backend:task_type" -> score
}

interface PrewarmData {
  alpha: number
  threshold: number
  scores: Record<string, number>  // "hour:tier" -> score
}

interface ScoringWeights {
  latency: number
  throughput: number
  reliability: number
  availability: number
  cost: number
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

interface RoutingConfig {
  prefer_local: boolean
  fallback_enabled: boolean
  load_balance: boolean
  scoring: ScoringWeights
  hedging: HedgingConfig
  prewarm: PrewarmConfig
}

interface SettingsConfig {
  routing: RoutingConfig
  [key: string]: unknown
}

async function loadSettings(): Promise<SettingsConfig | null> {
  try {
    const content = await readFile(SETTINGS_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    return null
  }
}

async function saveSettings(config: SettingsConfig): Promise<void> {
  await writeFile(SETTINGS_FILE, JSON.stringify(config, null, 2), "utf-8")
}

async function loadAffinityData(): Promise<AffinityData | null> {
  try {
    const content = await readFile(AFFINITY_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    return null
  }
}

async function loadPrewarmData(): Promise<PrewarmData | null> {
  try {
    const content = await readFile(PREWARM_FILE, "utf-8")
    return JSON.parse(content)
  } catch {
    return null
  }
}

function parseAffinityScores(scores: Record<string, number>): {
  byBackend: Record<string, Record<string, number>>
  topPairs: Array<{ backend: string; task: string; score: number }>
} {
  const byBackend: Record<string, Record<string, number>> = {}
  const topPairs: Array<{ backend: string; task: string; score: number }> = []

  for (const [key, score] of Object.entries(scores)) {
    const [backend, task] = key.split(":", 2)
    if (backend && task) {
      if (!byBackend[backend]) byBackend[backend] = {}
      byBackend[backend][task] = score
      topPairs.push({ backend, task, score })
    }
  }

  // Sort by score descending
  topPairs.sort((a, b) => b.score - a.score)

  return { byBackend, topPairs: topPairs.slice(0, 10) }
}

function parsePrewarmScores(scores: Record<string, number>, threshold: number): {
  byHour: Record<number, Record<string, number>>
  currentPredictions: string[]
  nextHourPredictions: string[]
} {
  const byHour: Record<number, Record<string, number>> = {}
  const currentHour = new Date().getHours()
  const nextHour = (currentHour + 1) % 24

  for (const [key, score] of Object.entries(scores)) {
    const [hourStr, tier] = key.split(":", 2)
    const hour = parseInt(hourStr)
    if (!isNaN(hour) && tier) {
      if (!byHour[hour]) byHour[hour] = {}
      byHour[hour][tier] = score
    }
  }

  // Get predictions above threshold
  const getPredictions = (hour: number) => {
    const hourScores = byHour[hour] || {}
    return Object.entries(hourScores)
      .filter(([, score]) => score >= threshold)
      .sort((a, b) => b[1] - a[1])
      .map(([tier]) => tier)
  }

  return {
    byHour,
    currentPredictions: getPredictions(currentHour),
    nextHourPredictions: getPredictions(nextHour),
  }
}

export async function GET() {
  try {
    const [settings, affinity, prewarm] = await Promise.all([
      loadSettings(),
      loadAffinityData(),
      loadPrewarmData(),
    ])

    if (!settings) {
      return NextResponse.json({
        error: "Settings not found",
      }, { status: 404 })
    }

    const routing = settings.routing || {}
    const scoringWeights = routing.scoring || {
      latency: 0.35,
      throughput: 0.15,
      reliability: 0.35,
      availability: 0.15,
      cost: 0.0,
    }
    const hedgingConfig = routing.hedging || {
      enabled: false,
      delay_ms: 50,
      max_backends: 2,
    }
    const prewarmConfig = routing.prewarm || {
      enabled: false,
      threshold: 0.3,
      check_interval_minutes: 5,
    }

    // Parse affinity data
    const affinityStatus = affinity
      ? {
          alpha: affinity.alpha || 0.1,
          trackedPairs: Object.keys(affinity.scores || {}).length,
          ...parseAffinityScores(affinity.scores || {}),
        }
      : {
          alpha: 0.1,
          trackedPairs: 0,
          byBackend: {},
          topPairs: [],
        }

    // Parse prewarm data
    const prewarmStatus = prewarm
      ? {
          alpha: prewarm.alpha || 0.15,
          threshold: prewarm.threshold || 0.3,
          trackedEntries: Object.keys(prewarm.scores || {}).length,
          currentHour: new Date().getHours(),
          ...parsePrewarmScores(prewarm.scores || {}, prewarm.threshold || 0.3),
        }
      : {
          alpha: 0.15,
          threshold: 0.3,
          trackedEntries: 0,
          currentHour: new Date().getHours(),
          byHour: {},
          currentPredictions: [],
          nextHourPredictions: [],
        }

    return NextResponse.json({
      scoring: scoringWeights,
      hedging: hedgingConfig,
      prewarm: prewarmConfig,
      affinity: affinityStatus,
      prewarmStatus,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    return NextResponse.json({
      error: "Failed to load routing data",
      details: error instanceof Error ? error.message : "Unknown error",
    }, { status: 500 })
  }
}

// PATCH: Update routing settings
export async function PATCH(request: Request) {
  try {
    const body = await request.json()
    const settings = await loadSettings()

    if (!settings) {
      return NextResponse.json({
        success: false,
        error: "Settings not found",
      }, { status: 404 })
    }

    // Update hedging config
    if (body.hedging) {
      settings.routing.hedging = {
        ...settings.routing.hedging,
        ...body.hedging,
      }
    }

    // Update prewarm config
    if (body.prewarm) {
      settings.routing.prewarm = {
        ...settings.routing.prewarm,
        ...body.prewarm,
      }
    }

    // Update scoring weights
    if (body.scoring) {
      settings.routing.scoring = {
        ...settings.routing.scoring,
        ...body.scoring,
      }
    }

    await saveSettings(settings)

    return NextResponse.json({
      success: true,
      message: "Routing settings updated",
      routing: settings.routing,
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    }, { status: 500 })
  }
}
