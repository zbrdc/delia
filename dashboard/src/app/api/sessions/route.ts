/**
 * Copyright (C) 2024 Delia Contributors
 * GPL-3.0-or-later
 */

import { NextResponse } from "next/server"
import { readdir, readFile, unlink } from "fs/promises"
import { join } from "path"
import { getSessionsDir } from "@/lib/paths"

interface SessionMessage {
  role: "user" | "assistant" | "system"
  content: string
  timestamp: string
  tokens?: number
  model?: string
}

interface SessionState {
  session_id: string
  created_at: string
  last_accessed: string
  messages: SessionMessage[]
  total_tokens: number
  total_calls: number
  original_task?: string
}

interface SessionSummary {
  id: string
  created_at: string
  last_accessed: string
  message_count: number
  total_tokens: number
  preview: string
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const sessionId = searchParams.get("id")

  try {
    const sessionsDir = getSessionsDir()

    if (sessionId) {
      // Get specific session
      const sessionFile = join(sessionsDir, `${sessionId}.json`)
      const data = await readFile(sessionFile, "utf-8")
      const session: SessionState = JSON.parse(data)
      return NextResponse.json({ success: true, session })
    }

    // List all sessions
    const files = await readdir(sessionsDir).catch(() => [])
    const sessions: SessionSummary[] = []

    for (const file of files) {
      if (!file.endsWith(".json")) continue
      try {
        const data = await readFile(join(sessionsDir, file), "utf-8")
        const session: SessionState = JSON.parse(data)
        const lastUserMsg = session.messages?.filter(m => m.role === "user").pop()
        sessions.push({
          id: session.session_id,
          created_at: session.created_at,
          last_accessed: session.last_accessed,
          message_count: session.messages?.length || 0,
          total_tokens: session.total_tokens || 0,
          preview: lastUserMsg?.content?.slice(0, 100) || session.original_task?.slice(0, 100) || "No messages",
        })
      } catch {
        // Skip invalid session files
      }
    }

    // Sort by last_accessed descending
    sessions.sort((a, b) => new Date(b.last_accessed).getTime() - new Date(a.last_accessed).getTime())

    return NextResponse.json({
      success: true,
      sessions,
      count: sessions.length,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ success: true, sessions: [], count: 0 })
    }
    return NextResponse.json({ success: false, error: "Failed to read sessions" }, { status: 500 })
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url)
  const sessionId = searchParams.get("id")

  if (!sessionId) {
    return NextResponse.json({ success: false, error: "Session ID required" }, { status: 400 })
  }

  try {
    const sessionFile = join(getSessionsDir(), `${sessionId}.json`)
    await unlink(sessionFile)
    return NextResponse.json({ success: true, deleted: sessionId })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ success: false, error: "Session not found" }, { status: 404 })
    }
    return NextResponse.json({ success: false, error: "Failed to delete session" }, { status: 500 })
  }
}
