import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

// Path to the circuit breaker file (in cache directory)
const CACHE_DIR = join(require("os").homedir(), ".cache", "delia")
const CIRCUIT_BREAKER_FILE = join(CACHE_DIR, "circuit_breaker.json")

interface CircuitBreakerStatus {
  available: boolean
  consecutive_failures: number
  last_error: string
  circuit_open: boolean
  seconds_until_available: number
  safe_context_kb: number
}

interface CircuitBreakerData {
  ollama: CircuitBreakerStatus
  llamacpp: CircuitBreakerStatus
  active_backend: string
  timestamp: string
}

export async function GET() {
  try {
    const data = await readFile(CIRCUIT_BREAKER_FILE, "utf-8")
    const circuitBreaker: CircuitBreakerData = JSON.parse(data)
    
    return NextResponse.json({
      ...circuitBreaker,
      success: true,
    })
  } catch (error) {
    // Return default status if file doesn't exist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({
        ollama: {
          available: true,
          consecutive_failures: 0,
          last_error: "",
          circuit_open: false,
          seconds_until_available: 0,
          safe_context_kb: 100,
        },
        llamacpp: {
          available: true,
          consecutive_failures: 0,
          last_error: "",
          circuit_open: false,
          seconds_until_available: 0,
          safe_context_kb: 100,
        },
        active_backend: "ollama",
        timestamp: new Date().toISOString(),
        success: true,
      })
    }
    
    return NextResponse.json(
      { error: "Failed to read circuit breaker status", success: false },
      { status: 500 }
    )
  }
}
