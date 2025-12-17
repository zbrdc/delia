/**
 * ChatView component for interactive chat sessions.
 *
 * Provides a REPL-style interface for multi-turn conversations.
 * Supports orchestrated mode with multi-model tool calling.
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useApp } from "ink";
import Spinner from "ink-spinner";
import { DeliaClient, type ChatEvent, type StatusEvent, type ToolCallInfo } from "../lib/api.js";
import { MarkdownText } from "./MarkdownText.js";
import { InputPrompt } from "./InputPrompt.js";
import { StatusIndicator, type StatusInfo } from "./StatusIndicator.js";

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

export interface ChatViewProps {
  /** API base URL (required) */
  apiUrl: string;
  /** Model to use */
  model?: string;
  /** Backend type preference */
  backendType?: string;
  /** Session ID to resume */
  sessionId?: string;
  /** 
   * Enable simple mode (no orchestration).
   * Default is FALSE - NLP orchestration is enabled by default.
   */
  simple?: boolean;
  /** 
   * Enable legacy tool-based orchestrated mode.
   * If true, uses old tool-calling approach instead of NLP.
   */
  legacyOrchestrated?: boolean;
  /** Include file/web tools in orchestrated mode */
  includeFileTools?: boolean;
  /** Workspace path for file operations */
  workspace?: string;
  /** Called when chat exits */
  onExit?: () => void;
}

const MessageBubble: React.FC<{
  message: Message;
  isLast?: boolean;
}> = ({ message, isLast }) => {
  const isUser = message.role === "user";

  // Build stats string for right side
  const stats: string[] = [];
  if (message.tokens !== undefined && message.tokens > 0) {
    stats.push(`${message.tokens} tok`);
  }
  if (message.elapsed_ms !== undefined) {
    stats.push(`${(message.elapsed_ms / 1000).toFixed(1)}s`);
  }
  const statsStr = stats.join(" · ");

  return (
    <Box flexDirection="column" marginY={1}>
      {/* Role header with stats on right */}
      <Box justifyContent="space-between">
        <Box>
          <Text color={isUser ? "blue" : "green"} bold>
            {isUser ? "You" : "Delia"}
          </Text>
          {message.model && (
            <Text color="dim"> ({message.model})</Text>
          )}
          {message.orchestrated && (
            <Text color="magenta"> [orchestrated]</Text>
          )}
        </Box>
        {statsStr && (
          <Text color="dim">{statsStr}</Text>
        )}
      </Box>

      {/* Tools used indicator */}
      {message.toolsUsed && message.toolsUsed.length > 0 && (
        <Box marginLeft={2}>
          <Text color="cyan">🔧 Tools: {message.toolsUsed.join(", ")}</Text>
        </Box>
      )}

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

/** Component to display tool calls as they happen */
const ToolCallsIndicator: React.FC<{ calls: ToolCallInfo[] }> = ({ calls }) => (
  <Box flexDirection="column" marginY={1} marginLeft={2}>
    {calls.map((call, i) => (
      <Box key={i}>
        <Text color="cyan">🔧 {call.name}</Text>
        <Text color="dim"> ({JSON.stringify(call.args).slice(0, 50)}...)</Text>
      </Box>
    ))}
  </Box>
);

/** Component to show real-time tool execution */
const ToolExecutionPanel: React.FC<{
  iteration: number;
  calls: ToolCallInfo[];
  results: Array<{ name: string; success: boolean; output_preview: string }>;
}> = ({ iteration, calls, results }) => (
  <Box flexDirection="column" marginY={1} borderStyle="single" borderColor="cyan" paddingX={1}>
    <Text color="cyan" bold>🔧 Tool Execution (Iteration {iteration})</Text>
    {calls.map((call, i) => {
      const result = results.find(r => r.name === call.name);
      return (
        <Box key={i} flexDirection="column" marginLeft={1}>
          <Box>
            <Text color={result?.success ? "green" : result ? "red" : "yellow"}>
              {result ? (result.success ? "✓" : "✗") : "⋯"} {call.name}
            </Text>
          </Box>
          {result && (
            <Box marginLeft={2}>
              <Text color="dim" wrap="truncate-end">
                {result.output_preview.slice(0, 100)}
              </Text>
            </Box>
          )}
        </Box>
      );
    })}
  </Box>
);

const ThinkingIndicator: React.FC<{ status: string }> = ({ status }) => (
  <Box marginY={1}>
    <Text color="yellow">
      <Spinner type="dots" /> {status}
    </Text>
  </Box>
);

const WelcomeMessage: React.FC<{ sessionId?: string; simple?: boolean }> = ({ sessionId, simple }) => (
  <Box flexDirection="column" marginY={1} paddingX={1}>
    <Box paddingX={2} paddingY={1}>
      <Text bold color="cyan">
        Delia
      </Text>
      <Text color="dim"> - Multi-Model AI Orchestrator</Text>
    </Box>
    <Box marginTop={1}>
      <Text color="dim">
        Commands: /help, /clear, /new, /simple (toggle mode)
      </Text>
    </Box>
    {simple ? (
      <Box>
        <Text color="dim">
          Simple mode - single model, no orchestration
        </Text>
      </Box>
    ) : (
      <Box flexDirection="column">
        <Text color="magenta">
          NLP Orchestration: voting, comparison, deep thinking
        </Text>
        <Text color="dim">
          Delia detects intent and orchestrates automatically
        </Text>
        <Text color="dim">
          Try: "verify this", "compare models", "think deeply"
        </Text>
      </Box>
    )}
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
  apiUrl,
  model,
  backendType,
  sessionId: initialSessionId,
  simple: initialSimple = false,  // NLP orchestration by DEFAULT!
  legacyOrchestrated = false,
  includeFileTools = false,
  workspace,
  onExit,
}) => {
  const { exit } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId);
  const [isProcessing, setIsProcessing] = useState(false);
  const [thinkingStatus, setThinkingStatus] = useState<string | null>(null);
  const [statusInfo, setStatusInfo] = useState<StatusInfo | null>(null);
  const [statusHistory, setStatusHistory] = useState<StatusInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string>("");
  const [simple, setSimple] = useState(initialSimple);
  const [currentToolCalls, setCurrentToolCalls] = useState<ToolCallInfo[]>([]);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [toolResults, setToolResults] = useState<Array<{ name: string; success: boolean; output_preview: string }>>([]);

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
      if (cmd === "/simple" || cmd === "/orchestrate" || cmd === "/orch") {
        setSimple((prev) => !prev);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: simple 
              ? "NLP Orchestration enabled - Delia auto-detects voting/comparison needs"
              : "Simple mode - Single model, no orchestration",
            timestamp: new Date(),
          },
        ]);
        return;
      }
      if (cmd === "/help") {
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: `Commands:
/clear - Clear messages
/new - Start new session  
/simple - Toggle simple/NLP mode
/help - Show this help
/exit - Exit chat

${!simple ? `NLP Orchestration (DEFAULT):
Delia detects your intent and orchestrates automatically:

- "verify this" / "make sure" -> K-voting for reliability
- "compare models" / "what do different models think" -> Multi-model comparison
- "think carefully" / "analyze thoroughly" -> Deep reasoning mode

Models receive role-specific prompts (code_reviewer, architect, etc.)
No tool calls shown - just natural conversation.` : "Simple mode - direct single-model chat, no orchestration"}`,
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
    setStatusHistory([]);
    setCurrentToolCalls([]);
    setCurrentIteration(0);
    setToolResults([]);

    // Track response in local variable to avoid closure issues
    let responseContent = "";
    let assistantModel: string | undefined;
    let elapsedMs: number | undefined;
    let tokens: number | undefined;
    let toolsUsed: string[] = [];
    let wasOrchestrated = false;
    let hasError = false;

    try {
      await client.sendChat(
        {
          message: input,
          sessionId,
          model,
          backendType,
          simple,  // NLP orchestration by default (when simple=false)
          orchestrated: legacyOrchestrated,  // Legacy tool-based mode
          includeFileTools,
          workspace,
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
              const newStatus: StatusInfo = {
                phase: event.phase as StatusInfo["phase"],
                message: event.message,
                details: event.details,
              };
              setStatusInfo(newStatus);
              // Keep status history so messages persist after completion
              setStatusHistory((prev) => [...prev, newStatus]);
              break;

            case "tools":
              // Summary of all tool calls (at end)
              setCurrentToolCalls(event.calls);
              break;

            case "tool_call":
              // Real-time tool call notification
              setCurrentIteration(event.iteration);
              setCurrentToolCalls(event.calls);
              setToolResults([]); // Reset results for new iteration
              break;

            case "tool_result":
              // Real-time tool result
              setToolResults((prev) => [...prev, {
                name: event.name,
                success: event.success,
                output_preview: event.output_preview,
              }]);
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
              tokens = event.tokens;
              toolsUsed = event.tools_used || [];
              wasOrchestrated = event.orchestrated || false;
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
            tokens: tokens,
            toolsUsed: toolsUsed.length > 0 ? toolsUsed : undefined,
            orchestrated: wasOrchestrated,
          },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsProcessing(false);
      setThinkingStatus(null);
      setStatusInfo(null);
      setCurrentToolCalls([]);
      setCurrentIteration(0);
      setToolResults([]);
      // Keep statusHistory - don't clear it so messages remain visible
      setCurrentResponse("");
    }
  }, [client, sessionId, model, backendType, simple, legacyOrchestrated, includeFileTools, workspace]);

  const handleExit = useCallback(() => {
    onExit?.();
    exit();
  }, [exit, onExit]);

  // Get last few messages to display (keep view manageable)
  const displayMessages = messages.slice(-20);

  return (
    <Box flexDirection="column" padding={1}>
      {/* Welcome/header */}
      {messages.length === 0 && <WelcomeMessage sessionId={sessionId} simple={simple} />}

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
            {!simple && (
              <Text color="magenta"> [nlp]</Text>
            )}
          </Box>
          <Box marginLeft={2}>
            <MarkdownText>{currentResponse}</MarkdownText>
          </Box>
        </Box>
      )}

      {/* Real-time tool execution panel (orchestrated mode) */}
      {isProcessing && currentIteration > 0 && currentToolCalls.length > 0 && (
        <ToolExecutionPanel
          iteration={currentIteration}
          calls={currentToolCalls}
          results={toolResults}
        />
      )}

      {/* Status indicators for advanced logic (routing, voting, quality) */}
      {/* Show history when not processing, or current status when processing */}
      {!isProcessing && statusHistory.length > 0 && (
        <Box flexDirection="column">
          {statusHistory.map((status, i) => (
            <StatusIndicator key={i} status={status} />
          ))}
        </Box>
      )}
      {isProcessing && statusInfo && <StatusIndicator status={statusInfo} />}

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
        prompt={simple ? "> " : ">> "}
        onSubmit={handleSubmit}
        onExit={handleExit}
        disabled={isProcessing}
        history={history}
        promptColor={simple ? "dim" : "cyan"}
      />
    </Box>
  );
};

export default ChatView;
