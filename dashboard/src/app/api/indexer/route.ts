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
import { readFile, stat } from "fs/promises"
import { join } from "path"

const DATA_DIR = join(process.cwd(), "..", "data")
const SUMMARY_FILE = join(DATA_DIR, "project_summary.json")
const GRAPH_FILE = join(DATA_DIR, "symbol_graph.json")

export async function GET() {
  try {
    let summaryData: Record<string, unknown> = {}
    let graphData: Record<string, unknown> = {}
    let lastIndexed = null

    try {
      const summaryRaw = await readFile(SUMMARY_FILE, "utf-8")
      summaryData = JSON.parse(summaryRaw)
      const summaryStat = await stat(SUMMARY_FILE)
      lastIndexed = summaryStat.mtime.toISOString()
    } catch (e) {
      console.warn("Could not read project_summary.json", e)
    }

    try {
      const graphRaw = await readFile(GRAPH_FILE, "utf-8")
      graphData = JSON.parse(graphRaw)
    } catch (e) {
      console.warn("Could not read symbol_graph.json", e)
    }

    const files = Object.keys(summaryData)
    const withEmbeddings = Object.values(summaryData).filter((s: any) => !!s.embedding).length
    
    let totalSymbols = 0
    Object.values(graphData).forEach((node: any) => {
      if (node.symbols) totalSymbols += node.symbols.length
    })

    const topSymbols = []
    const paths = Object.keys(graphData)
    for (const path of paths) {
      const node: any = graphData[path]
      if (node.symbols) {
        for (const sym of node.symbols) {
          topSymbols.push({ ...sym, file: path })
          if (topSymbols.length >= 20) break
        }
      }
      if (topSymbols.length >= 20) break
    }

    return NextResponse.json({
      success: true,
      stats: {
        totalFiles: files.length,
        withEmbeddings,
        totalSymbols,
        lastIndexed
      },
      topSymbols,
      files: files.slice(0, 50).map(path => ({
        path,
        hasEmbedding: !!(summaryData as any)[path].embedding,
        mtime: (summaryData as any)[path].mtime
      }))
    })
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: (error as Error).message,
    }, { status: 500 })
  }
}
