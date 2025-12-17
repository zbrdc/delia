/**
 * Streaming response component for Delia CLI.
 *
 * Renders agent responses with real-time updates using Ink.
 */

import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { MarkdownText } from "./MarkdownText.js";
import { ToolCallList, type ToolCall } from "./ToolCallPanel.js";
import { StatusIndicator, type StatusInfo } from "./StatusIndicator.js";

export type { ToolCall, StatusInfo };

export interface StreamingResponseProps {
  task: string;
  thinking: string | null;
  statusInfo?: StatusInfo | null;
  statusHistory?: StatusInfo[];
  response: string;
  toolCalls: ToolCall[];
  done: boolean;
  success: boolean;
  model?: string;
  backend?: string;
  elapsed_ms?: number;
  iterations?: number;
}

const StatusBar: React.FC<{
  model?: string;
  backend?: string;
  elapsed_ms?: number;
  iterations?: number;
  toolCalls: ToolCall[];
}> = ({ model, backend, elapsed_ms, iterations, toolCalls }) => {
  const toolNames = [...new Set(toolCalls.map((t) => t.name))].join(", ");

  const parts: string[] = [];
  if (model) parts.push(`Model: ${model}`);
  if (backend) parts.push(`Backend: ${backend}`);
  if (elapsed_ms !== undefined) parts.push(`${(elapsed_ms / 1000).toFixed(1)}s`);
  if (iterations !== undefined) parts.push(`${iterations} iteration${iterations !== 1 ? "s" : ""}`);
  if (toolNames) parts.push(`Tools: ${toolNames}`);

  return (
    <Box
      flexDirection="row"
      marginTop={1}
      paddingX={1}
      borderStyle="single"
      borderColor="dim"
    >
      <Text color="dim">{parts.join(" | ")}</Text>
    </Box>
  );
};

export const StreamingResponse: React.FC<StreamingResponseProps> = ({
  task,
  thinking,
  statusInfo,
  statusHistory = [],
  response,
  toolCalls,
  done,
  success,
  model,
  backend,
  elapsed_ms,
  iterations,
}) => {
  return (
    <Box flexDirection="column">
      {/* Task header */}
      <Box
        borderStyle="round"
        borderColor="blue"
        paddingX={2}
        paddingY={1}
        marginBottom={1}
      >
        <Text bold color="blue">
          Task:{" "}
        </Text>
        <Text>{task}</Text>
      </Box>

      {/* Status indicators for advanced logic (routing, voting, quality) */}
      {/* Show history when done, or current status when processing */}
      {done && statusHistory.length > 0 && (
        <Box flexDirection="column">
          {statusHistory.map((status, i) => (
            <StatusIndicator key={i} status={status} />
          ))}
        </Box>
      )}
      {!done && statusInfo && <StatusIndicator status={statusInfo} />}

      {/* Thinking indicator */}
      {thinking && !done && (
        <Box marginY={1}>
          <Text color="dim">
            <Spinner type="dots" /> {thinking}
          </Text>
        </Box>
      )}

      {/* Tool calls */}
      <ToolCallList toolCalls={toolCalls} />

      {/* Response */}
      {response && (
        <Box marginY={1}>
          <MarkdownText>{response}</MarkdownText>
        </Box>
      )}

      {/* Status bar */}
      {done && (
        <StatusBar
          model={model}
          backend={backend}
          elapsed_ms={elapsed_ms}
          iterations={iterations}
          toolCalls={toolCalls}
        />
      )}

      {/* Completion indicator */}
      {done && (
        <Box marginTop={1}>
          <Text color={success ? "green" : "yellow"}>
            {success ? "[OK]" : "[WARN]"} Agent{" "}
            {success ? "completed" : "completed with warnings"}
          </Text>
        </Box>
      )}
    </Box>
  );
};
