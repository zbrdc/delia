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
import { getCircuitBreakerFile } from "@/lib/paths"

// Use same path resolution as CLI
const CIRCUIT_BREAKER_FILE = getCircuitBreakerFile()

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
