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
import { Panel, Divider, Chip } from "./Panel.js";
import { ThinkingIndicator, IterationCounter, TokenCounter } from "./ProgressBar.js";

export type { ToolCall, StatusInfo };

export interface StreamingResponseProps {
  task: string;
  thinking: string | null;
  statusInfo?: StatusInfo | null;
  response: string;
  toolCalls: ToolCall[];
  done: boolean;
  success: boolean;
  model?: string;
  backend?: string;
  elapsed_ms?: number;
  iterations?: number;
  tokens?: number;
}

/**
 * Task header panel.
 */
const TaskHeader: React.FC<{
  task: string;
  model?: string;
  backend?: string;
}> = ({ task, model, backend }) => (
  <Panel
    title="Task"
    icon="🎯"
    titleColor="blue"
    borderStyle="round"
    borderColor="blue"
    headerRight={
      <Box>
        {model && <Text color="dim">{model}</Text>}
        {backend && <Text color="dim"> @ {backend}</Text>}
      </Box>
    }
  >
    <Text>{task}</Text>
  </Panel>
);

/**
 * Stats bar shown at completion.
 */
const StatsBar: React.FC<{
  model?: string;
  backend?: string;
  elapsed_ms?: number;
  iterations?: number;
  tokens?: number;
  toolCalls: ToolCall[];
}> = ({ model, backend, elapsed_ms, iterations, tokens, toolCalls }) => {
  const toolNames = [...new Set(toolCalls.map((t) => t.name))];
  const successCount = toolCalls.filter((t) => t.status === "success").length;
  const failedCount = toolCalls.filter((t) => t.status === "error").length;

  return (
    <Box
      flexDirection="row"
      marginTop={1}
      paddingX={2}
      paddingY={1}
      borderStyle="single"
      borderColor="dim"
      justifyContent="space-between"
    >
      {/* Left side - model/backend info */}
      <Box>
        {model && (
          <>
            <Text color="cyan">🤖 {model}</Text>
            {backend && <Text color="dim"> @ {backend}</Text>}
          </>
        )}
      </Box>

      {/* Center - timing and iterations */}
      <Box>
        {elapsed_ms !== undefined && (
          <Text color="dim">⏱️ {(elapsed_ms / 1000).toFixed(1)}s</Text>
        )}
        {iterations !== undefined && (
          <>
            <Text color="dim"> · </Text>
            <Text color="dim">{iterations} iter</Text>
          </>
        )}
        {tokens !== undefined && tokens > 0 && (
          <>
            <Text color="dim"> · </Text>
            <TokenCounter tokens={tokens} elapsed_ms={elapsed_ms} showRate={false} />
          </>
        )}
      </Box>

      {/* Right side - tool summary */}
      <Box>
        {toolNames.length > 0 && (
          <>
            <Text color="cyan">🔧 </Text>
            {successCount > 0 && <Text color="green">✓{successCount}</Text>}
            {failedCount > 0 && <Text color="red"> ✗{failedCount}</Text>}
          </>
        )}
      </Box>
    </Box>
  );
};

/**
 * Completion indicator with status.
 */
const CompletionIndicator: React.FC<{
  success: boolean;
  elapsed_ms?: number;
}> = ({ success, elapsed_ms }) => (
  <Box marginTop={1}>
    <Text color={success ? "green" : "yellow"} bold>
      {success ? "✓" : "⚠"} Agent {success ? "completed" : "completed with warnings"}
    </Text>
    {elapsed_ms !== undefined && (
      <Text color="dim"> in {(elapsed_ms / 1000).toFixed(1)}s</Text>
    )}
  </Box>
);

export const StreamingResponse: React.FC<StreamingResponseProps> = ({
  task,
  thinking,
  statusInfo,
  response,
  toolCalls,
  done,
  success,
  model,
  backend,
  elapsed_ms,
  iterations,
  tokens,
}) => {
  return (
    <Box flexDirection="column">
      {/* Task header */}
      <TaskHeader task={task} model={model} backend={backend} />

      {/* Status indicator - only shown while processing, disappears when done */}
      {!done && statusInfo && (
        <StatusIndicator status={statusInfo} />
      )}

      {/* Thinking indicator */}
      {thinking && !done && (
        <ThinkingIndicator status={thinking} variant="melon" />
      )}

      {/* Iteration counter */}
      {!done && iterations !== undefined && iterations > 0 && (
        <Box marginY={1}>
          <IterationCounter current={iterations} label="Iteration" />
        </Box>
      )}

      {/* Tool calls */}
      <ToolCallList toolCalls={toolCalls} />

      {/* Separator before response */}
      {response && <Divider text="Response" />}

      {/* Response */}
      {response && (
        <Panel
          borderStyle="round"
          borderColor={done ? "green" : "yellow"}
          padding={1}
        >
          <MarkdownText>{response}</MarkdownText>
        </Panel>
      )}

      {/* Processing indicator when not done and no response yet */}
      {!done && !response && !thinking && (
        <Box marginY={1}>
          <Text color="yellow">
            <Spinner type="dots" /> Processing...
          </Text>
        </Box>
      )}

      {/* Status bar */}
      {done && (
        <StatsBar
          model={model}
          backend={backend}
          elapsed_ms={elapsed_ms}
          iterations={iterations}
          tokens={tokens}
          toolCalls={toolCalls}
        />
      )}

      {/* Completion indicator */}
      {done && <CompletionIndicator success={success} elapsed_ms={elapsed_ms} />}
    </Box>
  );
};

/**
 * Simple response panel for non-streaming responses.
 */
export const ResponsePanel: React.FC<{
  content: string;
  model?: string;
  elapsed_ms?: number;
  tokens?: number;
}> = ({ content, model, elapsed_ms, tokens }) => (
  <Panel
    title="Response"
    icon="🍈"
    titleColor="green"
    borderStyle="round"
    borderColor="green"
    headerRight={
      <Box>
        {model && <Text color="dim">{model}</Text>}
        {elapsed_ms !== undefined && (
          <Text color="dim"> · {(elapsed_ms / 1000).toFixed(1)}s</Text>
        )}
        {tokens !== undefined && (
          <Text color="dim"> · {tokens} tok</Text>
        )}
      </Box>
    }
  >
    <MarkdownText>{content}</MarkdownText>
  </Panel>
);
