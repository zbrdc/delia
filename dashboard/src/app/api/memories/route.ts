/**
 * Copyright (C) 2024 Delia Contributors
 * GPL-3.0-or-later
 */

import { NextResponse } from "next/server"
import { readdir, readFile, writeFile, unlink, stat } from "fs/promises"
import { join } from "path"
import { getMemoriesDir } from "@/lib/paths"

interface MemorySummary {
  name: string
  size: number
  path: string
  modified: string
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const memoryName = searchParams.get("name")

  try {
    const memoriesDir = getMemoriesDir()

    if (memoryName) {
      // Get specific memory content
      const memoryFile = join(memoriesDir, `${memoryName}.md`)
      const content = await readFile(memoryFile, "utf-8")
      const stats = await stat(memoryFile)

      return NextResponse.json({
        success: true,
        name: memoryName,
        content,
        size: stats.size,
        modified: stats.mtime.toISOString(),
      })
    }

    // List all memories
    const files = await readdir(memoriesDir).catch(() => [])
    const memories: MemorySummary[] = []

    for (const file of files) {
      if (!file.endsWith(".md")) continue
      try {
        const filePath = join(memoriesDir, file)
        const stats = await stat(filePath)
        memories.push({
          name: file.replace(".md", ""),
          size: stats.size,
          path: filePath,
          modified: stats.mtime.toISOString(),
        })
      } catch {
        // Skip invalid files
      }
    }

    // Sort by modified descending
    memories.sort((a, b) => new Date(b.modified).getTime() - new Date(a.modified).getTime())

    return NextResponse.json({
      success: true,
      memories,
      count: memories.length,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ success: true, memories: [], count: 0 })
    }
    return NextResponse.json({ success: false, error: "Failed to read memories" }, { status: 500 })
  }
}

export async function PUT(request: Request) {
  try {
    const body = await request.json()
    const { name, content } = body

    if (!name || content === undefined) {
      return NextResponse.json({ success: false, error: "Name and content required" }, { status: 400 })
    }

    const memoryFile = join(getMemoriesDir(), `${name}.md`)
    await writeFile(memoryFile, content, "utf-8")

    return NextResponse.json({ success: true, updated: name })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to update memory" }, { status: 500 })
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { name, content } = body

    if (!name) {
      return NextResponse.json({ success: false, error: "Name required" }, { status: 400 })
    }

    // Sanitize name
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "_")
    const memoryFile = join(getMemoriesDir(), `${safeName}.md`)

    // Check if exists
    try {
      await stat(memoryFile)
      return NextResponse.json({ success: false, error: "Memory already exists" }, { status: 409 })
    } catch {
      // File doesn't exist, good
    }

    await writeFile(memoryFile, content || `# ${safeName}\n\n`, "utf-8")

    return NextResponse.json({ success: true, created: safeName })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to create memory" }, { status: 500 })
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url)
  const name = searchParams.get("name")

  if (!name) {
    return NextResponse.json({ success: false, error: "Name required" }, { status: 400 })
  }

  try {
    const memoryFile = join(getMemoriesDir(), `${name}.md`)
    await unlink(memoryFile)
    return NextResponse.json({ success: true, deleted: name })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ success: false, error: "Memory not found" }, { status: 404 })
    }
    return NextResponse.json({ success: false, error: "Failed to delete memory" }, { status: 500 })
  }
}
