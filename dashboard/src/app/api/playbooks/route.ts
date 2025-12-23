/**
 * Copyright (C) 2024 Delia Contributors
 * GPL-3.0-or-later
 */

import { NextResponse } from "next/server"
import { readdir, readFile, writeFile, unlink } from "fs/promises"
import { join } from "path"
import { getPlaybooksDir } from "@/lib/paths"

interface PlaybookBullet {
  id: string
  content: string
  section?: string
  helpful_count: number
  harmful_count: number
  created_at: string
  last_used?: string
  utility_score: number
}

interface PlaybookSummary {
  name: string
  bullet_count: number
  path: string
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const playbookName = searchParams.get("name")

  try {
    const playbooksDir = getPlaybooksDir()

    if (playbookName) {
      // Get specific playbook
      const playbookFile = join(playbooksDir, `${playbookName}.json`)
      const data = await readFile(playbookFile, "utf-8")
      let bullets: PlaybookBullet[] = JSON.parse(data)

      // Handle wrapped format
      if (!Array.isArray(bullets) && bullets && typeof bullets === "object") {
        bullets = (bullets as { bullets?: PlaybookBullet[] }).bullets || []
      }

      // Calculate utility scores
      bullets = bullets.map(b => ({
        ...b,
        utility_score: b.helpful_count + b.harmful_count > 0
          ? b.helpful_count / (b.helpful_count + b.harmful_count)
          : 0.5,
      }))

      return NextResponse.json({
        success: true,
        name: playbookName,
        bullets,
        count: bullets.length,
      })
    }

    // List all playbooks
    const files = await readdir(playbooksDir).catch(() => [])
    const playbooks: PlaybookSummary[] = []

    for (const file of files) {
      if (!file.endsWith(".json")) continue
      try {
        const data = await readFile(join(playbooksDir, file), "utf-8")
        let bullets = JSON.parse(data)
        if (!Array.isArray(bullets) && bullets?.bullets) {
          bullets = bullets.bullets
        }
        playbooks.push({
          name: file.replace(".json", ""),
          bullet_count: Array.isArray(bullets) ? bullets.length : 0,
          path: join(playbooksDir, file),
        })
      } catch {
        // Skip invalid files
      }
    }

    // Sort by name
    playbooks.sort((a, b) => a.name.localeCompare(b.name))

    return NextResponse.json({
      success: true,
      playbooks,
      count: playbooks.length,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json({ success: true, playbooks: [], count: 0 })
    }
    return NextResponse.json({ success: false, error: "Failed to read playbooks" }, { status: 500 })
  }
}

export async function PUT(request: Request) {
  try {
    const body = await request.json()
    const { name, bullet_id, content } = body

    if (!name || !bullet_id) {
      return NextResponse.json({ success: false, error: "Name and bullet_id required" }, { status: 400 })
    }

    const playbookFile = join(getPlaybooksDir(), `${name}.json`)
    const data = await readFile(playbookFile, "utf-8")
    let bullets: PlaybookBullet[] = JSON.parse(data)

    if (!Array.isArray(bullets) && bullets && typeof bullets === "object") {
      bullets = (bullets as { bullets?: PlaybookBullet[] }).bullets || []
    }

    const idx = bullets.findIndex(b => b.id === bullet_id)
    if (idx === -1) {
      return NextResponse.json({ success: false, error: "Bullet not found" }, { status: 404 })
    }

    bullets[idx].content = content
    bullets[idx].last_used = new Date().toISOString()

    await writeFile(playbookFile, JSON.stringify(bullets, null, 2))

    return NextResponse.json({ success: true, updated: bullet_id })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to update bullet" }, { status: 500 })
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { name, content, section } = body

    if (!name || !content) {
      return NextResponse.json({ success: false, error: "Name and content required" }, { status: 400 })
    }

    const playbookFile = join(getPlaybooksDir(), `${name}.json`)
    let bullets: PlaybookBullet[] = []

    try {
      const data = await readFile(playbookFile, "utf-8")
      bullets = JSON.parse(data)
      if (!Array.isArray(bullets) && bullets && typeof bullets === "object") {
        bullets = (bullets as { bullets?: PlaybookBullet[] }).bullets || []
      }
    } catch {
      // File doesn't exist, start fresh
    }

    const newBullet: PlaybookBullet = {
      id: `strat-${Math.random().toString(16).slice(2, 10)}`,
      content,
      section: section || "general",
      helpful_count: 0,
      harmful_count: 0,
      created_at: new Date().toISOString(),
      utility_score: 0.5,
    }

    bullets.push(newBullet)
    await writeFile(playbookFile, JSON.stringify(bullets, null, 2))

    return NextResponse.json({ success: true, bullet: newBullet })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to add bullet" }, { status: 500 })
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url)
  const name = searchParams.get("name")
  const bulletId = searchParams.get("bullet_id")

  if (!name || !bulletId) {
    return NextResponse.json({ success: false, error: "Name and bullet_id required" }, { status: 400 })
  }

  try {
    const playbookFile = join(getPlaybooksDir(), `${name}.json`)
    const data = await readFile(playbookFile, "utf-8")
    let bullets: PlaybookBullet[] = JSON.parse(data)

    if (!Array.isArray(bullets) && bullets && typeof bullets === "object") {
      bullets = (bullets as { bullets?: PlaybookBullet[] }).bullets || []
    }

    const idx = bullets.findIndex(b => b.id === bulletId)
    if (idx === -1) {
      return NextResponse.json({ success: false, error: "Bullet not found" }, { status: 404 })
    }

    bullets.splice(idx, 1)
    await writeFile(playbookFile, JSON.stringify(bullets, null, 2))

    return NextResponse.json({ success: true, deleted: bulletId })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to delete bullet" }, { status: 500 })
  }
}
