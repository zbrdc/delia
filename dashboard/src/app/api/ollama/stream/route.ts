/**
 * Copyright (C) 2023 the project owner
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
import { NextRequest } from "next/server"

// Ollama API endpoint
const OLLAMA_BASE = process.env.OLLAMA_BASE || "http://localhost:11434"

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const type = searchParams.get("type") || "ps" // ps = running models, logs = generation stream

  if (type === "ps") {
    // Get currently running models
    try {
      const response = await fetch(`${OLLAMA_BASE}/api/ps`)
      const data = await response.json()
      return Response.json(data)
    } catch {
      return Response.json({ error: "Failed to connect to Ollama", models: [] }, { status: 500 })
    }
  }

  return Response.json({ error: "Unknown type" }, { status: 400 })
}

// SSE endpoint for streaming Ollama logs
export async function POST(request: NextRequest) {
  const body = await request.json()
  const { prompt, model } = body

  if (!model) {
    return Response.json({ error: "Model is required" }, { status: 400 })
  }
  const encoder = new TextEncoder()
  
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const response = await fetch(`${OLLAMA_BASE}/api/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt,
            stream: true,
          }),
        })

        if (!response.ok || !response.body) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: "Failed to connect to Ollama" })}\n\n`))
          controller.close()
          return
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split("\n").filter(Boolean)

          for (const line of lines) {
            try {
              const data = JSON.parse(line)
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`))
            } catch {
              // Skip invalid JSON
            }
          }
        }

        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`))
        controller.close()
      } catch (error) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: String(error) })}\n\n`))
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  })
}
