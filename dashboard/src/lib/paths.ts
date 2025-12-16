/**
 * Delia Path Configuration for Dashboard
 *
 * Mirrors the path resolution logic from src/delia/paths.py
 * to ensure CLI and dashboard use the same config/data files.
 */

import { join } from "path"
import { homedir } from "os"
import { existsSync } from "fs"

// Project root (dashboard is in {repo}/dashboard, so go up one level)
const PROJECT_ROOT = join(process.cwd(), "..")

// User config root
const USER_DELIA_DIR = join(homedir(), ".delia")

/**
 * Find settings file with same priority as CLI:
 * 1. DELIA_SETTINGS_FILE environment variable
 * 2. settings.json in Current Working Directory
 * 3. settings.json in ~/.delia/ (global user config)
 * 4. settings.json in Project Root (legacy/dev mode)
 */
export function getSettingsFile(): string {
  const envPath = process.env.DELIA_SETTINGS_FILE
  if (envPath && existsSync(envPath)) {
    return envPath
  }

  const cwdPath = join(process.cwd(), "settings.json")
  if (existsSync(cwdPath)) {
    return cwdPath
  }

  const userPath = join(USER_DELIA_DIR, "settings.json")
  if (existsSync(userPath)) {
    return userPath
  }

  // Fallback to project root (create here if needed)
  return join(PROJECT_ROOT, "settings.json")
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

  // If running from source/dev and data exists, use it
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

// Export constants for convenience
export const SETTINGS_FILE = getSettingsFile()
export const DATA_DIR = getDataDir()
export const CACHE_DIR = getCacheDir()
export const LIVE_LOGS_FILE = getLogsFile()
