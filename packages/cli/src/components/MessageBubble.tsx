/**
 * MessageBubble component for chat messages.
 *
 * Provides visually distinct message bubbles for user, assistant, and system messages.
 */

import React from "react";
import { Box, Text } from "ink";
import { MarkdownText } from "./MarkdownText.js";
import { Badge, Chip } from "./Panel.js";

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  elapsed_ms?: number;
  tokens?: number;
  toolsUsed?: string[];
  orchestrated?: boolean;
}

export interface MessageBubbleProps {
  message: Message;
  isLast?: boolean;
  /** Whether to show full content or truncate */
  expanded?: boolean;
  /** Whether message is currently streaming */
  streaming?: boolean;
}

// Role configurations
const ROLE_CONFIG = {
  user: {
    color: "blue",
    label: "You",
    icon: "",
    borderColor: "blue",
    align: "right" as const,
  },
  assistant: {
    color: "green",
    label: "Delia",
    icon: "🍈",
    borderColor: "green",
    align: "left" as const,
  },
  system: {
    color: "yellow",
    label: "System",
    icon: "🍈",
    borderColor: "dim",
    align: "center" as const,
  },
};

/**
 * Format timestamp for display.
 */
function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/**
 * User message bubble - right-aligned styling.
 */
const UserBubble: React.FC<{ message: Message }> = ({ message }) => {
  return (
    <Box flexDirection="column" marginY={1} alignItems="flex-end">
      {/* Header */}
      <Box marginBottom={1}>
        <Text color="dim">{formatTime(message.timestamp)} · </Text>
        <Text color="blue" bold>
          {ROLE_CONFIG.user.label}
        </Text>
      </Box>

      {/* Content with right border */}
      <Box
        borderStyle="single"
        borderColor="blue"
        borderRight
        borderLeft={false}
        borderTop={false}
        borderBottom={false}
        paddingRight={2}
      >
        <Text>{message.content}</Text>
      </Box>
    </Box>
  );
};

/**
 * Extract short model name (remove prefix/version details).
 */
function shortModelName(model: string): string {
  // Remove common prefixes and extract core name
  // e.g. "qwen2.5-coder:14b-instruct-q8_0" -> "qwen2.5-coder:14b"
  const cleaned = model
    .replace(/-instruct.*$/i, "")
    .replace(/-q\d+_?\d*$/i, "")
    .replace(/:latest$/i, "");
  return cleaned;
}

/**
 * Assistant message bubble - clean, minimal styling.
 */
const AssistantBubble: React.FC<{
  message: Message;
  streaming?: boolean;
}> = ({ message, streaming }) => {
  // Simple timing display
  const timeStr = message.elapsed_ms !== undefined
    ? `${(message.elapsed_ms / 1000).toFixed(1)}s`
    : null;

  return (
    <Box flexDirection="column" marginY={1}>
      {/* Header row - simplified */}
      <Box justifyContent="space-between" marginBottom={1}>
        <Box>
          <Text color="green" bold>
            {ROLE_CONFIG.assistant.icon} {ROLE_CONFIG.assistant.label}
          </Text>
          {message.model && (
            <Text color="dim"> {shortModelName(message.model)}</Text>
          )}
          {message.orchestrated && (
            <Text color="magenta"> ✨</Text>
          )}
          {streaming && (
            <Text color="yellow"> ◐</Text>
          )}
        </Box>
        {timeStr && (
          <Text color="dim">{timeStr}</Text>
        )}
      </Box>

      {/* Content */}
      <Box paddingLeft={2}>
        <MarkdownText>{message.content}</MarkdownText>
      </Box>
    </Box>
  );
};

/**
 * System message bubble - minimal centered styling.
 */
const SystemBubble: React.FC<{ message: Message }> = ({ message }) => {
  return (
    <Box marginY={1} justifyContent="center">
      <Box
        paddingX={2}
        borderStyle="single"
        borderColor="dim"
      >
        <Text color="dim" italic>
          {ROLE_CONFIG.system.icon} {message.content}
        </Text>
      </Box>
    </Box>
  );
};

/**
 * Main MessageBubble component.
 */
export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isLast,
  expanded = true,
  streaming = false,
}) => {
  switch (message.role) {
    case "user":
      return <UserBubble message={message} />;
    case "assistant":
      return <AssistantBubble message={message} streaming={streaming} />;
    case "system":
      return <SystemBubble message={message} />;
  }
};

/**
 * Streaming message indicator - shown while response is being generated.
 */
export interface StreamingBubbleProps {
  content: string;
  orchestrated?: boolean;
}

export const StreamingBubble: React.FC<StreamingBubbleProps> = ({
  content,
  orchestrated,
}) => {
  return (
    <Box flexDirection="column" marginY={1}>
      {/* Header */}
      <Box marginBottom={1}>
        <Text color="green" bold>
          {ROLE_CONFIG.assistant.icon} {ROLE_CONFIG.assistant.label}
        </Text>
        {orchestrated && (
          <Text color="magenta" bold>
            {" "}
            ✨
          </Text>
        )}
        <Text color="yellow"> ◐</Text>
      </Box>

      {/* Content */}
      <Box paddingLeft={2}>
        {content ? (
          <MarkdownText>{content}</MarkdownText>
        ) : (
          <Text color="dim" italic>
            thinking...
          </Text>
        )}
      </Box>
    </Box>
  );
};

export default MessageBubble;
