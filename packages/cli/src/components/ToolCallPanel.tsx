/**
 * ToolCallPanel component for displaying tool calls with expandable output.
 *
 * Features:
 * - Status indicators (running, success, error)
 * - Collapsible output sections
 * - Syntax highlighting for common output types
 * - Diff display for file modifications
 * - Animated progress for running tools
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import Spinner from "ink-spinner";
import { highlight, supportsLanguage } from "cli-highlight";
import { DiffViewer } from "./DiffViewer.js";
import { Panel } from "./Panel.js";

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  status: "running" | "success" | "error";
  output?: string;
  elapsed_ms?: number;
}

interface ToolCallPanelProps {
  tool: ToolCall;
  index: number;
  isLast: boolean;
  defaultExpanded?: boolean;
}

/**
 * Detect output type and apply appropriate highlighting.
 */
function detectAndHighlight(output: string): { highlighted: string; type: string } {
  // Check for diff output
  if (output.includes("@@") && (output.includes("---") || output.includes("+++"))) {
    return { highlighted: output, type: "diff" };
  }

  // Check for JSON
  if (output.startsWith("{") || output.startsWith("[")) {
    try {
      JSON.parse(output);
      return {
        highlighted: highlight(output, { language: "json", ignoreIllegals: true }),
        type: "json",
      };
    } catch {
      // Not valid JSON
    }
  }

  // Check for Python
  if (output.includes("def ") || output.includes("class ") || output.includes("import ")) {
    return {
      highlighted: highlight(output, { language: "python", ignoreIllegals: true }),
      type: "python",
    };
  }

  // Check for JavaScript/TypeScript
  if (
    output.includes("function ") ||
    output.includes("const ") ||
    output.includes("=>") ||
    output.includes("export ")
  ) {
    return {
      highlighted: highlight(output, { language: "typescript", ignoreIllegals: true }),
      type: "typescript",
    };
  }

  // Check for shell output (file listings, command output)
  if (output.match(/^(total \d+|drwx|[-r][-w][-x])/m)) {
    return {
      highlighted: highlight(output, { language: "bash", ignoreIllegals: true }),
      type: "shell",
    };
  }

  return { highlighted: output, type: "text" };
}

/**
 * Format tool arguments for display.
 */
function formatArgs(args: Record<string, unknown>, maxLength = 60): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";

  const parts: string[] = [];
  for (const [key, value] of entries) {
    const valueStr = typeof value === "string" ? value : JSON.stringify(value);
    const truncated = valueStr.length > 40 ? valueStr.slice(0, 37) + "..." : valueStr;
    parts.push(`${key}=${truncated}`);
  }

  const result = parts.join(", ");
  return result.length > maxLength ? result.slice(0, maxLength - 3) + "..." : result;
}

/**
 * Get icon and color for tool status.
 */
function getStatusIndicator(status: ToolCall["status"]): { icon: string; color: string } {
  switch (status) {
    case "running":
      return { icon: "", color: "yellow" };
    case "success":
      return { icon: "✓", color: "green" };
    case "error":
      return { icon: "✗", color: "red" };
  }
}

/**
 * Truncate output for collapsed view.
 */
function truncateOutput(output: string, maxLines = 3): { text: string; truncated: boolean } {
  const lines = output.split("\n");
  if (lines.length <= maxLines) {
    return { text: output, truncated: false };
  }
  return {
    text: lines.slice(0, maxLines).join("\n"),
    truncated: true,
  };
}

/**
 * Output viewer with syntax detection and truncation.
 */
const OutputViewer: React.FC<{
  output: string;
  expanded: boolean;
  maxCollapsedLines?: number;
  maxExpandedLines?: number;
}> = ({ output, expanded, maxCollapsedLines = 3, maxExpandedLines = 30 }) => {
  const { highlighted, type } = detectAndHighlight(output);

  // Handle diff output specially
  if (type === "diff") {
    return (
      <Box marginLeft={2} marginY={1}>
        <DiffViewer diff={output} maxLines={expanded ? 50 : 10} showLineNumbers={expanded} />
      </Box>
    );
  }

  const maxLines = expanded ? maxExpandedLines : maxCollapsedLines;
  const { text, truncated } = truncateOutput(highlighted, maxLines);

  return (
    <Box flexDirection="column" marginLeft={2}>
      <Box paddingX={2}>
        <Text>{text}</Text>
      </Box>
      {truncated && !expanded && (
        <Text color="dim" italic>
          {" "}
          ... (press Enter to expand)
        </Text>
      )}
      {truncated && expanded && (
        <Text color="dim" italic>
          {" "}
          ... ({output.split("\n").length - maxExpandedLines} more lines)
        </Text>
      )}
    </Box>
  );
};

/**
 * Tool icon based on tool name.
 */
const ToolIcon: React.FC<{ name: string }> = ({ name }) => {
  const icons: Record<string, string> = {
    read_file: "📄",
    list_directory: "📁",
    search_code: "🔍",
    web_fetch: "🌐",
    delegate: "🤖",
    think: "🧠",
    compare: "⚖️",
    vote: "🗳️",
    default: "🔧",
  };

  // Match partial names
  for (const [key, icon] of Object.entries(icons)) {
    if (name.toLowerCase().includes(key.replace("_", ""))) {
      return <Text>{icon}</Text>;
    }
  }

  return <Text>{icons.default}</Text>;
};

/**
 * Individual tool call panel with expandable output.
 */
export const ToolCallPanel: React.FC<ToolCallPanelProps> = ({
  tool,
  index,
  isLast,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const { icon, color } = getStatusIndicator(tool.status);
  const argsStr = formatArgs(tool.args);

  // Interactive expansion (only for completed tools with output)
  const canExpand = tool.status !== "running" && tool.output && tool.output.length > 100;

  return (
    <Box
      flexDirection="column"
      marginBottom={isLast ? 0 : 1}
      borderStyle={tool.status === "running" ? "round" : "single"}
      borderColor={color}
      paddingX={1}
      paddingY={0}
    >
      {/* Header row */}
      <Box justifyContent="space-between">
        <Box>
          {/* Status indicator */}
          {tool.status === "running" ? (
            <Text color={color}>
              <Spinner type="dots" />{" "}
            </Text>
          ) : (
            <Text color={color}>{icon} </Text>
          )}

          {/* Tool icon and name */}
          <ToolIcon name={tool.name} />
          <Text bold color="yellow">
            {" "}{tool.name}
          </Text>

          {/* Timing */}
          {tool.elapsed_ms !== undefined && (
            <Text color="dim"> ({tool.elapsed_ms}ms)</Text>
          )}
        </Box>

        {/* Expand indicator */}
        {canExpand && (
          <Text color="dim">[{expanded ? "▼" : "▶"}]</Text>
        )}
      </Box>

      {/* Arguments row */}
      {argsStr && (
        <Box marginLeft={3}>
          <Text color="dim">{argsStr}</Text>
        </Box>
      )}

      {/* Output (if exists and not running) */}
      {tool.output && tool.status !== "running" && (
        <OutputViewer output={tool.output} expanded={expanded || !canExpand} />
      )}

      {/* Running indicator */}
      {tool.status === "running" && (
        <Box marginLeft={3}>
          <Text color="dim" italic>
            executing...
          </Text>
        </Box>
      )}

      {/* Error message for failed tools */}
      {tool.status === "error" && !tool.output && (
        <Box marginLeft={3}>
          <Text color="red" italic>
            Tool execution failed
          </Text>
        </Box>
      )}
    </Box>
  );
};

/**
 * Container for multiple tool calls with summary.
 */
export const ToolCallList: React.FC<{
  toolCalls: ToolCall[];
  showSummary?: boolean;
}> = ({ toolCalls, showSummary = true }) => {
  if (toolCalls.length === 0) return null;

  const completed = toolCalls.filter((t) => t.status === "success").length;
  const failed = toolCalls.filter((t) => t.status === "error").length;
  const running = toolCalls.filter((t) => t.status === "running").length;

  return (
    <Box flexDirection="column" marginY={1}>
      {/* Summary header */}
      {showSummary && toolCalls.length > 1 && (
        <Box
          marginBottom={1}
          paddingX={2}
          borderStyle="single"
          borderColor="dim"
        >
          <Text color="cyan" bold>🔧 Tools </Text>
          <Text color="dim">│ </Text>
          {completed > 0 && <Text color="green">✓ {completed} </Text>}
          {failed > 0 && <Text color="red">✗ {failed} </Text>}
          {running > 0 && (
            <Text color="yellow">
              <Spinner type="dots" /> {running}
            </Text>
          )}
        </Box>
      )}

      {/* Individual tool panels */}
      {toolCalls.map((tool, i) => (
        <ToolCallPanel
          key={i}
          tool={tool}
          index={i}
          isLast={i === toolCalls.length - 1}
        />
      ))}
    </Box>
  );
};

/**
 * Real-time tool execution panel for orchestrated mode.
 * Simplified - no iteration counter, minimal chrome.
 */
export interface ToolExecutionInfo {
  name: string;
  args?: Record<string, unknown>;
  success?: boolean;
  output_preview?: string;
}

export const ToolExecutionPanel: React.FC<{
  calls: ToolExecutionInfo[];
  results: ToolExecutionInfo[];
}> = ({ calls, results }) => {
  if (calls.length === 0) return null;

  return (
    <Box flexDirection="column" marginY={1}>
      {calls.map((call, i) => {
        const result = results.find((r) => r.name === call.name);
        const isRunning = !result;
        const isSuccess = result?.success;

        return (
          <Box key={i}>
            {/* Status */}
            {isRunning ? (
              <Text color="yellow">
                <Spinner type="dots" />{" "}
              </Text>
            ) : isSuccess ? (
              <Text color="green">✓ </Text>
            ) : (
              <Text color="red">✗ </Text>
            )}

            {/* Tool name */}
            <ToolIcon name={call.name} />
            <Text color={isRunning ? "yellow" : "dim"}>
              {" "}{call.name}
            </Text>
          </Box>
        );
      })}
    </Box>
  );
};

export default ToolCallPanel;
