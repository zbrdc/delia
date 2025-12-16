/**
 * ChatView component for interactive chat sessions.
 *
 * Provides a REPL-style interface for multi-turn conversations.
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useApp } from "ink";
import Spinner from "ink-spinner";
import { DeliaClient, type ChatEvent, type StatusEvent } from "../lib/api.js";
import { MarkdownText } from "./MarkdownText.js";
import { InputPrompt } from "./InputPrompt.js";
import { StatusIndicator, type StatusInfo } from "./StatusIndicator.js";

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  elapsed_ms?: number;
}

export interface ChatViewProps {
  /** API base URL */
  apiUrl?: string;
  /** Model to use */
  model?: string;
  /** Backend type preference */
  backendType?: string;
  /** Session ID to resume */
  sessionId?: string;
  /** Called when chat exits */
  onExit?: () => void;
}

const MessageBubble: React.FC<{
  message: Message;
  isLast?: boolean;
}> = ({ message, isLast }) => {
  const isUser = message.role === "user";

  return (
    <Box flexDirection="column" marginY={1}>
      {/* Role header */}
      <Box>
        <Text color={isUser ? "blue" : "green"} bold>
          {isUser ? "You" : "Delia"}
        </Text>
        {message.model && (
          <Text color="dim"> ({message.model})</Text>
        )}
        {message.elapsed_ms !== undefined && (
          <Text color="dim"> - {(message.elapsed_ms / 1000).toFixed(1)}s</Text>
        )}
      </Box>

      {/* Message content */}
      <Box marginLeft={2}>
        {isUser ? (
          <Text>{message.content}</Text>
        ) : (
          <MarkdownText>{message.content}</MarkdownText>
        )}
      </Box>
    </Box>
  );
};

const ThinkingIndicator: React.FC<{ status: string }> = ({ status }) => (
  <Box marginY={1}>
    <Text color="yellow">
      <Spinner type="dots" /> {status}
    </Text>
  </Box>
);

const WelcomeMessage: React.FC<{ sessionId?: string }> = ({ sessionId }) => (
  <Box flexDirection="column" marginY={1} paddingX={1}>
    <Box borderStyle="round" borderColor="cyan" paddingX={2} paddingY={1}>
      <Text bold color="cyan">
        Delia Chat
      </Text>
    </Box>
    <Box marginTop={1}>
      <Text color="dim">
        Type your message and press Enter. Commands: /exit, /clear, /new
      </Text>
    </Box>
    {sessionId && (
      <Box>
        <Text color="dim">
          Session: {sessionId.slice(0, 8)}...
        </Text>
      </Box>
    )}
  </Box>
);

export const ChatView: React.FC<ChatViewProps> = ({
  apiUrl = "http://localhost:8201",
  model,
  backendType,
  sessionId: initialSessionId,
  onExit,
}) => {
  const { exit } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId);
  const [isProcessing, setIsProcessing] = useState(false);
  const [thinkingStatus, setThinkingStatus] = useState<string | null>(null);
  const [statusInfo, setStatusInfo] = useState<StatusInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string>("");

  const client = new DeliaClient(apiUrl);

  const handleSubmit = useCallback(async (input: string) => {
    // Handle commands
    if (input.startsWith("/")) {
      const cmd = input.toLowerCase();
      if (cmd === "/clear") {
        setMessages([]);
        setError(null);
        return;
      }
      if (cmd === "/new") {
        setMessages([]);
        setSessionId(undefined);
        setError(null);
        return;
      }
      if (cmd === "/help") {
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: "Commands:\n- /exit - Exit chat\n- /clear - Clear messages\n- /new - Start new session\n- /help - Show this help",
            timestamp: new Date(),
          },
        ]);
        return;
      }
    }

    // Add user message
    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setHistory((prev) => [...prev, input]);
    setIsProcessing(true);
    setError(null);
    setCurrentResponse("");
    setStatusInfo(null);

    // Track response in local variable to avoid closure issues
    let responseContent = "";
    let assistantModel: string | undefined;
    let elapsedMs: number | undefined;
    let hasError = false;

    try {
      await client.sendChat(
        {
          message: input,
          sessionId,
          model,
          backendType,
        },
        (event: ChatEvent) => {
          switch (event.type) {
            case "session":
              if (event.created) {
                setSessionId(event.id);
              }
              break;

            case "thinking":
              setThinkingStatus(event.status);
              break;

            case "status":
              // Advanced logic status (routing, voting, quality)
              setStatusInfo({
                phase: event.phase,
                message: event.message,
                details: event.details,
              });
              break;

            case "token":
              responseContent += event.content;
              setCurrentResponse(responseContent);
              break;

            case "response":
              responseContent = event.content;
              setCurrentResponse(event.content);
              break;

            case "error":
              hasError = true;
              setError(event.message);
              break;

            case "done":
              assistantModel = event.model;
              elapsedMs = event.elapsed_ms;
              break;
          }
        }
      );

      // Add assistant message using local variable
      if (responseContent && !hasError) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: responseContent,
            timestamp: new Date(),
            model: assistantModel,
            elapsed_ms: elapsedMs,
          },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsProcessing(false);
      setThinkingStatus(null);
      setStatusInfo(null);
      setCurrentResponse("");
    }
  }, [client, sessionId, model, backendType]);

  const handleExit = useCallback(() => {
    onExit?.();
    exit();
  }, [exit, onExit]);

  // Get last few messages to display (keep view manageable)
  const displayMessages = messages.slice(-20);

  return (
    <Box flexDirection="column" padding={1}>
      {/* Welcome/header */}
      {messages.length === 0 && <WelcomeMessage sessionId={sessionId} />}

      {/* Messages */}
      {displayMessages.map((msg, i) => (
        <MessageBubble
          key={i}
          message={msg}
          isLast={i === displayMessages.length - 1}
        />
      ))}

      {/* Current streaming response */}
      {currentResponse && (
        <Box flexDirection="column" marginY={1}>
          <Box>
            <Text color="green" bold>
              Delia
            </Text>
          </Box>
          <Box marginLeft={2}>
            <MarkdownText>{currentResponse}</MarkdownText>
          </Box>
        </Box>
      )}

      {/* Status indicator for advanced logic (routing, voting, quality) */}
      {statusInfo && <StatusIndicator status={statusInfo} />}

      {/* Thinking indicator */}
      {thinkingStatus && <ThinkingIndicator status={thinkingStatus} />}

      {/* Error display */}
      {error && (
        <Box marginY={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      {/* Separator */}
      <Box marginY={1}>
        <Text color="dim">{"─".repeat(60)}</Text>
      </Box>

      {/* Input prompt */}
      <InputPrompt
        prompt="you: "
        onSubmit={handleSubmit}
        onExit={handleExit}
        disabled={isProcessing}
        history={history}
        promptColor="blue"
      />
    </Box>
  );
};

export default ChatView;
