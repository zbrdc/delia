/**
 * Copyright (C) 2024 Delia Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { getSymbolGraphFile, getProjectSummaryFile } from "@/lib/paths"

interface GraphNode {
  id: string
  symbols: number
  imports: string[]
  group: string  // directory group for coloring
  summary?: string
}

interface GraphLink {
  source: string
  target: string
}

export async function GET() {
  try {
    const graphRaw = await readFile(getSymbolGraphFile(), "utf-8")
    const graphData = JSON.parse(graphRaw)

    // Try to load summaries for richer tooltips
    let summaries: Record<string, { summary?: string }> = {}
    try {
      const summaryRaw = await readFile(getProjectSummaryFile(), "utf-8")
      summaries = JSON.parse(summaryRaw)
    } catch {
      // Summaries optional
    }

    const nodes: GraphNode[] = []
    const links: GraphLink[] = []
    const nodeIds = new Set<string>()

    // Build nodes - filter to only Python source files
    for (const [path, node] of Object.entries(graphData)) {
      // Only include Python files in src/delia/
      if (!path.endsWith(".py")) continue
      if (!path.startsWith("src/delia/")) continue
      // Skip test files and __pycache__
      if (path.includes("__pycache__") || path.includes("/tests/")) continue

      const fileNode = node as { symbols?: unknown[]; imports?: string[] }
      const dir = path.split("/").slice(0, -1).join("/") || "root"

      nodes.push({
        id: path,
        symbols: fileNode.symbols?.length || 0,
        imports: fileNode.imports || [],
        group: dir,
        summary: summaries[path]?.summary || undefined
      })
      nodeIds.add(path)
    }

    // Build links from imports
    for (const node of nodes) {
      for (const imp of node.imports) {
        // Check if import resolves to a known file
        // Handle relative imports (REL:level:module)
        let targetPath: string | null = null

        if (imp.startsWith("REL:")) {
          const [, levelStr, moduleName] = imp.split(":", 3)
          const level = parseInt(levelStr, 10)
          const parts = node.id.split("/")
          const parentParts = parts.slice(0, -(level))

          if (moduleName) {
            const candidates = [
              [...parentParts, moduleName + ".py"].join("/"),
              [...parentParts, moduleName, "__init__.py"].join("/"),
            ]
            for (const c of candidates) {
              if (nodeIds.has(c)) {
                targetPath = c
                break
              }
            }
          } else {
            const initPath = [...parentParts, "__init__.py"].join("/")
            if (nodeIds.has(initPath)) targetPath = initPath
          }
        } else {
          // Absolute import - try common patterns
          const pyPath = imp.replace(/\./g, "/") + ".py"
          const initPath = imp.replace(/\./g, "/") + "/__init__.py"

          if (nodeIds.has(pyPath)) targetPath = pyPath
          else if (nodeIds.has(initPath)) targetPath = initPath
          else if (nodeIds.has("src/" + pyPath)) targetPath = "src/" + pyPath
          else if (nodeIds.has("src/delia/" + pyPath.replace("delia/", ""))) {
            targetPath = "src/delia/" + pyPath.replace("delia/", "")
          }
        }

        if (targetPath && targetPath !== node.id) {
          links.push({ source: node.id, target: targetPath })
        }
      }
    }

    // Get unique directory groups for legend
    const groups = [...new Set(nodes.map(n => n.group))].sort()

    return NextResponse.json({
      success: true,
      nodes,
      links,
      groups,
      stats: {
        totalNodes: nodes.length,
        totalLinks: links.length,
        totalGroups: groups.length
      }
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: (error as Error).message,
      nodes: [],
      links: [],
      groups: []
    }, { status: 500 })
  }
}
