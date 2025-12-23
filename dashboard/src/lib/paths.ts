/**
 * Delia Path Configuration for Dashboard
 *
 * Mirrors the path resolution logic from src/delia/paths.py
 * to ensure CLI and dashboard use the same config/data files.
 */

import { join, dirname } from "path"
import { homedir } from "os"
import { existsSync } from "fs"
import { fileURLToPath } from "url"

// Get the actual project root by going up from this file's location
// This file is at: dashboard/src/lib/paths.ts
// Project root is: dashboard/../ = repo root
function getProjectRoot(): string {
  // MCP server passes project root via environment variable
  const envRoot = process.env.DELIA_PROJECT_ROOT
  if (envRoot && existsSync(envRoot)) {
    return envRoot
  }

  // In Next.js, __dirname may not work, but we can use import.meta or process.cwd
  // For reliability, check multiple locations
  const candidates = [
    // If running from dashboard directory (npm run dev)
    join(process.cwd(), ".."),
    // If running from repo root
    process.cwd(),
    // Relative to this file (for ESM)
    join(dirname(fileURLToPath(import.meta.url)), "..", "..", ".."),
  ]

  // Find the one that has a pyproject.toml (definitive marker of repo root)
  for (const candidate of candidates) {
    if (existsSync(join(candidate, "pyproject.toml"))) {
      return candidate
    }
  }

  // Fallback to dashboard parent
  return join(process.cwd(), "..")
}

const PROJECT_ROOT = getProjectRoot()

// User config root
const USER_DELIA_DIR = join(homedir(), ".delia")

/**
 * Find settings file with same priority as CLI:
 * 1. DELIA_SETTINGS_FILE environment variable
 * 2. settings.json in ~/.delia/ (global user config - preferred)
 * 3. settings.json in Project Root (dev mode)
 * 
 * Note: We skip CWD check for dashboard since CWD is unpredictable
 * (could be dashboard/, repo root, etc. depending on how it's run)
 */
export function getSettingsFile(): string {
  const envPath = process.env.DELIA_SETTINGS_FILE
  if (envPath && existsSync(envPath)) {
    return envPath
  }

  // User config takes priority (consistent location)
  const userPath = join(USER_DELIA_DIR, "settings.json")
  if (existsSync(userPath)) {
    return userPath
  }

  // Project root for dev mode
  const projectPath = join(PROJECT_ROOT, "settings.json")
  if (existsSync(projectPath)) {
    return projectPath
  }

  // Fallback - will be created in user dir
  return userPath
}

/**
 * Find data directory with same priority as CLI:
 * 1. DELIA_DATA_DIR environment variable
 * 2. Project Root data (if exists - legacy/dev compatibility)
 * 3. ~/.delia/data (global user data)
 */
export function getDataDir(): string {
  const envPath = process.env.DELIA_DATA_DIR
  if (envPath) {
    return envPath
  }

  // If running from source/dev and data dir exists, use it
  const projectData = join(PROJECT_ROOT, "data")
  if (existsSync(projectData)) {
    return projectData
  }

  return join(USER_DELIA_DIR, "data")
}

// Derived paths
export function getCacheDir(): string {
  return join(getDataDir(), "cache")
}

export function getLogsFile(): string {
  return join(getCacheDir(), "live_logs.json")
}

export function getStatsFile(): string {
  return join(getCacheDir(), "usage_stats.json")
}

export function getEnhancedStatsFile(): string {
  return join(getCacheDir(), "enhanced_stats.json")
}

export function getCircuitBreakerFile(): string {
  return join(getCacheDir(), "circuit_breaker.json")
}

export function getBackendMetricsFile(): string {
  return join(getCacheDir(), "backend_metrics.json")
}

export function getAffinityFile(): string {
  return join(getCacheDir(), "affinity.json")
}

export function getVotingStatsFile(): string {
  return join(getCacheDir(), "voting_stats.json")
}

export function getMelonsFile(): string {
  return join(getDataDir(), "melons.json")
}

// Project-specific paths (per-project .delia/ directory)
export function getProjectDeliaDir(): string {
  return join(PROJECT_ROOT, ".delia")
}

export function getSessionsDir(): string {
  return join(getProjectDeliaDir(), "sessions")
}

export function getPlaybooksDir(): string {
  return join(getProjectDeliaDir(), "playbooks")
}

export function getMemoriesDir(): string {
  return join(getProjectDeliaDir(), "memories")
}

export function getProjectSummaryFile(): string {
  return join(getProjectDeliaDir(), "project_summary.json")
}

export function getSymbolGraphFile(): string {
  return join(getProjectDeliaDir(), "symbol_graph.json")
}

// Export constants for convenience
export const SETTINGS_FILE = getSettingsFile()
export const DATA_DIR = getDataDir()
export const CACHE_DIR = getCacheDir()
export const LIVE_LOGS_FILE = getLogsFile()
export const PROJECT_DELIA_DIR = getProjectDeliaDir()
export const SESSIONS_DIR = getSessionsDir()
export const PLAYBOOKS_DIR = getPlaybooksDir()
export const MEMORIES_DIR = getMemoriesDir()
