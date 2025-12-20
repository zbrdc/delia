# ADR-002: MCP-Native Paradigm (No REST API)

## Status
Accepted

## Context
Delia routes prompts to local LLMs. We needed to decide between:
1. **REST API** - HTTP endpoints with OpenAPI spec
2. **MCP Protocol** - Model Context Protocol tools over STDIO/SSE
3. **Both** - REST wrapper around MCP tools

## Decision
Use MCP Protocol exclusively, no REST API layer.

## Rationale

### Why MCP-only
- **Purpose-built**: MCP is designed for LLM tool calling
- **Client ecosystem**: Claude, IDEs, and agents already speak MCP
- **Simpler stack**: No OpenAPI, no route handlers, no schema duplication
- **Transport flexibility**: Same tools work over STDIO (local) or SSE (remote)

### Why not REST
- **Wrong paradigm**: REST is request-response; LLM tools are streaming conversations
- **Redundant**: Would duplicate MCP tool definitions as HTTP endpoints
- **No audience**: LLMs don't call REST APIs, they call MCP tools

### HTTP when needed
FastMCP already provides HTTP/SSE transport for remote deployment:
```bash
delia serve --transport sse --port 8200
```

## Trade-offs
- **No Swagger UI**: Can't browse API in browser
- **No curl testing**: Must use MCP client to test tools
- **Integration**: External systems must use MCP client SDK

## Consequences
- Tools are defined once in `mcp_server.py`
- Documentation focuses on MCP tool schemas
- Testing uses `fastmcp.test_client` instead of HTTP
- Dashboard API is separate (Next.js API routes for internal use only)
