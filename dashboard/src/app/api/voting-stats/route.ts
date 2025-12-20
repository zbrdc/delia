/**
 * Copyright (C) 2024 Delia Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { getVotingStatsFile } from "@/lib/paths"

// Use same path resolution as CLI
const VOTING_STATS_FILE = getVotingStatsFile()

interface ConsensusStats {
  total_attempts: number
  successful: number
  failed: number
  consensus_rate: number
  avg_votes: number
  p50_votes: number
  p95_votes: number
  distribution: Record<string, number>
  votes_histogram?: Record<string, number>  // Raw histogram from persistence
  votes_samples?: number[]  // Rolling window samples
}

interface RejectionRecord {
  timestamp: number
  reason: string
  backend_id: string
  tier: string
  response_preview: string
}

interface RejectionStats {
  total: number
  by_reason: Record<string, number>
  by_backend: Record<string, number>
  by_tier: Record<string, number>
  recent: RejectionRecord[]
}

interface TierStats {
  calls: number
  avg_quality: number
  quality_ema: number
  rejection_rate: number
  consensus_rate: number
}

interface VotingStats {
  consensus: ConsensusStats
  rejections: RejectionStats
  tiers: Record<string, TierStats>
  last_updated?: number
}

export async function GET() {
  try {
    const data = await readFile(VOTING_STATS_FILE, "utf-8")
    const stats: VotingStats = JSON.parse(data)

    // Format the stats for dashboard display
    const formatted = {
      consensus: {
        total_attempts: stats.consensus?.total_attempts || 0,
        successful: stats.consensus?.successful || 0,
        failed: stats.consensus?.failed || 0,
        consensus_rate: calculateRate(stats.consensus?.successful, stats.consensus?.total_attempts),
        avg_votes: stats.consensus?.avg_votes || 0,
        p50_votes: stats.consensus?.p50_votes || 0,
        p95_votes: stats.consensus?.p95_votes || 0,
        distribution: formatDistribution(stats.consensus?.votes_histogram || {}),
      },
      rejections: {
        total: stats.rejections?.total || 0,
        by_reason: stats.rejections?.by_reason || {},
        by_backend: stats.rejections?.by_backend || {},
        by_tier: stats.rejections?.by_tier || {},
        recent: (stats.rejections?.recent || []).slice(-10).map(r => ({
          ...r,
          timestamp_formatted: new Date(r.timestamp * 1000).toLocaleTimeString(),
        })),
      },
      tiers: formatTiers(stats.tiers || {}),
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json(formatted)
  } catch (error) {
    // Return empty stats if file doesn't exist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        consensus: {
          total_attempts: 0,
          successful: 0,
          failed: 0,
          consensus_rate: 0,
          avg_votes: 0,
          p50_votes: 0,
          p95_votes: 0,
          distribution: {},
        },
        rejections: {
          total: 0,
          by_reason: {},
          by_backend: {},
          by_tier: {},
          recent: [],
        },
        tiers: {},
        timestamp: new Date().toISOString(),
      })
    }

    return NextResponse.json(
      { error: "Failed to read voting stats file" },
      { status: 500 }
    )
  }
}

function calculateRate(success: number | undefined, total: number | undefined): number {
  if (!total || total === 0) return 0
  return Math.round(((success || 0) / total) * 1000) / 10 // One decimal place
}

function formatDistribution(histogram: Record<string, number>): Record<string, number> {
  const total = Object.values(histogram).reduce((sum, count) => sum + count, 0)
  if (total === 0) return {}

  const result: Record<string, number> = {}
  for (const [k, count] of Object.entries(histogram)) {
    result[`k=${k}`] = Math.round((count / total) * 1000) / 10
  }
  return result
}

function formatTiers(tiers: Record<string, {
  calls?: number
  quality_sum?: number
  rejections?: number
  consensus_attempts?: number
  consensus_successes?: number
  quality_ema?: number
}>): Record<string, TierStats> {
  const result: Record<string, TierStats> = {}

  for (const [name, tier] of Object.entries(tiers)) {
    const calls = tier.calls || 0
    result[name] = {
      calls,
      avg_quality: calls > 0 ? Math.round(((tier.quality_sum || 0) / calls) * 1000) / 1000 : 0,
      quality_ema: Math.round((tier.quality_ema || 0.5) * 1000) / 1000,
      rejection_rate: calls > 0 ? Math.round(((tier.rejections || 0) / calls) * 1000) / 10 : 0,
      consensus_rate: (tier.consensus_attempts || 0) > 0
        ? Math.round(((tier.consensus_successes || 0) / (tier.consensus_attempts || 1)) * 1000) / 10
        : 0,
    }
  }

  return result
}
