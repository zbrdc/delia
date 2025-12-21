#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def visualize():
    data_dir = Path("../data")
    summary_file = data_dir / "project_summary.json"
    graph_file = data_dir / "symbol_graph.json"

    if not summary_file.exists():
        print(f"Error: {summary_file} not found.")
        return

    print("\n" + "="*60)
    print(" DELIA ARCHITECTURAL INDEX VISUALIZER ".center(60, "="))
    print("="*60)

    with open(summary_file, "r") as f:
        summaries = json.load(f)

    print(f"\n[Index Statistics]")
    print(f"  Total Indexed Files: {len(summaries)}")
    
    # Calculate vector stats
    with_vectors = [s for s in summaries.values() if s.get("embedding")]
    print(f"  Files with Embeddings: {len(with_vectors)} / {len(summaries)}")
    
    if with_vectors:
        vec_dim = len(with_vectors[0]["embedding"])
        print(f"  Vector Dimensions: {vec_dim} (mxbai-embed-large)")

    print(f"\n[File Breakdown]")
    # Sort files by directory for better viewing
    sorted_paths = sorted(summaries.keys())
    for path in sorted_paths:
        s = summaries[path]
        status = "✅" if s.get("embedding") else "❌"
        # Extract file extension
        ext = path.split(".")[-1] if "." in path else "none"
        print(f"  {status} {path:<40} [{ext:>4}]")

    if graph_file.exists():
        with open(graph_file, "r") as f:
            graph = json.load(f)
        
        print(f"\n[Symbol Graph Statistics]")
        total_symbols = 0
        # The graph is a dict of {path: {symbols: [], imports: [], mtime: 0}}
        for node in graph.values():
            total_symbols += len(node.get("symbols", []))
        
        print(f"  Tracked File Nodes: {len(graph)}")
        print(f"  Total Extracted Symbols: {total_symbols}")
        
        # Show top symbols
        print(f"\n[Key Symbols (First 10)]")
        count = 0
        for path, node in graph.items():
            for sym in node.get("symbols", []):
                if count >= 10: break
                print(f"  • {sym['name']:<25} ({sym['kind']:<10}) in {path}")
                count += 1
            if count >= 10: break

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    visualize()