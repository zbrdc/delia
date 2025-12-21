#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent / "src"))

async def test_awareness():
    print("\n" + "="*70)
    print(" TESTING DELIA ARCHITECTURAL AWARENESS ".center(70, "="))
    print("="*70)

    from delia.orchestration.context import ContextEngine
    from delia.orchestration.summarizer import get_summarizer
    from delia.orchestration.graph import get_symbol_graph
    
    # 1. Test Root Discovery
    summarizer = get_summarizer()
    graph = get_symbol_graph()
    
    await summarizer.initialize()
    await graph.initialize()
    
    print(f"\n[Project Context]")
    print(f"  Detected Root: {summarizer.root}")
    print(f"  Files in Index: {len(summarizer.summaries)}")
    print(f"  Nodes in Graph: {len(graph.nodes)}")

    # 2. Test Cross-Project Connectivity (GraphRAG)
    # Let's see what happens when we focus on a backend file with many dependencies
    test_file = "src/delia/orchestration/summarizer.py"
    print(f"\n[Testing GraphRAG for: {test_file}]")
    
    if test_file in graph.nodes:
        related = graph.get_related_files(test_file)
        print(f"  Found {len(related)} related files via Symbol Graph:")
        for rf in related:
            # Get summary if available
            sum_text = summarizer.summaries.get(rf, {}).summary if rf in summarizer.summaries else "No summary"
            print(f"  • {rf:<40}")
    else:
        print(f"  ❌ {test_file} not found in graph nodes.")

    # 3. Test Context Assembly
    print(f"\n[Testing Prompt Assembly]")
    # This simulates what is sent to the LLM
    final_prompt = await ContextEngine.prepare_content(
        content="How do the backend summaries get to this route?",
        files=test_file,
        include_project_overview=True
    )
    
    # Check if context from OTHER directories was injected
    print(f"  Final Prompt Size: {len(final_prompt)} chars")
    
    if "### Architectural Context (Dependency Graph)" in final_prompt:
        print("  ✅ Success: GraphRAG injected cross-file architectural context!")
        # Peek at the injected context
        start = final_prompt.find("### Architectural Context")
        end = final_prompt.find("### Project Context", start + 1)
        if end == -1: end = final_prompt.find("### Task:", start + 1)
        print("\n--- Injected Architectural Awareness ---")
        print(final_prompt[start:end].strip())
        print("\n----------------------------------------")
    else:
        print("  ❌ Failure: Cross-file context was not injected.")

    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(test_awareness())
