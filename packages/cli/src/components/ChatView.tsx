/**
 * ChatView component for interactive chat sessions.
 *
 * Provides a REPL-style interface for multi-turn conversations.
 * Supports orchestrated mode with multi-model tool calling.
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useApp } from "ink";
import Spinner from "ink-spinner";
import { DeliaClient, type ChatEvent } from "../lib/api.js";
import { MarkdownText } from "./MarkdownText.js";
import { InputPrompt } from "./InputPrompt.js";
import { StatusIndicator, VotingProgress, type StatusInfo } from "./StatusIndicator.js";
import { Header, WelcomeScreen } from "./Header.js";
import { MessageBubble, StreamingBubble, type Message } from "./MessageBubble.js";
import { Panel, Divider } from "./Panel.js";
import { ThinkingIndicator } from "./ProgressBar.js";
import { ConfirmPrompt } from "./ConfirmPrompt.js";

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
  /**
   * Include file/web tools in orchestrated mode.
   * Default is TRUE - web search and file tools available by default.
   */
  includeFileTools?: boolean;
  /** Workspace path for file operations */
  workspace?: string;
  /** Called when chat exits */
  onExit?: () => void;
}

/**
 * Help panel content.
 */
const HelpPanel: React.FC<{ simple: boolean }> = ({ simple }) => (
  <Panel title="Help" icon="❓" titleColor="cyan" borderColor="cyan">
    <Box flexDirection="column">
      <Text color="dim" bold>Commands:</Text>
      <Box marginLeft={2} flexDirection="column">
        <Text><Text color="cyan">/clear</Text> - Clear messages</Text>
        <Text><Text color="cyan">/new</Text> - Start new session</Text>
        <Text><Text color="cyan">/simple</Text> - Toggle simple/NLP mode</Text>
        <Text><Text color="cyan">/help</Text> - Show this help</Text>
        <Text><Text color="cyan">/exit</Text> - Exit chat</Text>
      </Box>

      {!simple && (
        <>
          <Box marginTop={1}>
            <Text color="magenta" bold>✨ NLP Orchestration (DEFAULT):</Text>
          </Box>
          <Box marginLeft={2} flexDirection="column">
            <Text color="dim">Delia detects your intent and orchestrates automatically:</Text>
            <Text>• "verify this" / "make sure" → K-voting for reliability</Text>
            <Text>• "compare models" / "what do different models think" → Multi-model</Text>
            <Text>• "think carefully" / "analyze thoroughly" → Deep reasoning</Text>
          </Box>
        </>
      )}
    </Box>
  </Panel>
);

export const ChatView: React.FC<ChatViewProps> = ({
  apiUrl,
  model,
  backendType,
  sessionId: initialSessionId,
  simple: initialSimple = false,  // NLP orchestration by DEFAULT!
  legacyOrchestrated = false,
  includeFileTools = true,  // Web search and file tools enabled by default
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
  const [error, setError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string>("");
  const [simple, setSimple] = useState(initialSimple);
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "connecting" | "disconnected" | "error">("connecting");
  const [showHelp, setShowHelp] = useState(false);
  const [pendingConfirmation, setPendingConfirmation] = useState<{
    confirmId: string;
    tool: string;
    args: Record<string, unknown>;
    message: string;
  } | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const client = new DeliaClient(apiUrl);

  // Check connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        setConnectionStatus("connecting");
        await client.health();
        setConnectionStatus("connected");
      } catch {
        setConnectionStatus("error");
      }
    };
    checkConnection();
  }, [apiUrl]);

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
              ? "✨ NLP Orchestration enabled - Delia auto-detects voting/comparison needs"
              : "📝 Simple mode - Single model, no orchestration",
            timestamp: new Date(),
          },
        ]);
        return;
      }
      if (cmd === "/help") {
        setShowHelp((prev) => !prev);
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
    setShowHelp(false);

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
              // Transient status (routing, voting, quality) - disappears when done
              setStatusInfo({
                phase: event.phase as StatusInfo["phase"],
                message: event.message,
                details: event.details,
              });
              break;

            case "confirm":
              // File operation confirmation prompt
              setPendingConfirmation({
                confirmId: event.confirm_id,
                tool: event.tool,
                args: event.args,
                message: event.message,
              });
              setThinkingStatus(null); // Clear thinking while waiting
              break;

            case "tools":
              // Tool calls completed - we just track for the message
              break;

            case "tool_call":
              // Tool execution in progress
              break;

            case "tool_result":
              // Real-time tool result - handled by status updates
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
      setCurrentResponse("");
    }
  }, [client, sessionId, model, backendType, simple, legacyOrchestrated, includeFileTools, workspace]);

  const handleExit = useCallback(() => {
    onExit?.();
    exit();
  }, [exit, onExit]);

  // Handle interrupt (Escape key during processing)
  const handleInterrupt = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
    setIsProcessing(false);
    setThinkingStatus(null);
    setStatusInfo(null);
    setPendingConfirmation(null);
    if (currentResponse) {
      // Keep partial response as message
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: currentResponse + "\n\n*[interrupted]*",
          timestamp: new Date(),
        },
      ]);
    }
    setCurrentResponse("");
  }, [abortController, currentResponse]);

  // Handle confirmation prompt responses
  const handleConfirm = useCallback(
    async (confirmId: string, confirmed: boolean, allowAll: boolean) => {
      setPendingConfirmation(null);
      try {
        await client.confirmTool({ confirmId, confirmed, allowAll });
      } catch (err) {
        setError(`Confirmation failed: ${err instanceof Error ? err.message : "Unknown error"}`);
      }
    },
    [client]
  );

  // Get last few messages to display (keep view manageable)
  const displayMessages = messages.slice(-20);

  // Determine current mode
  const currentMode = simple ? "simple" : legacyOrchestrated ? "orchestrated" : "nlp";

  return (
    <Box flexDirection="column" padding={1}>
      {/* Header - always visible */}
      <Header
        variant={messages.length === 0 ? "compact" : "minimal"}
        sessionId={sessionId}
        status={connectionStatus}
        mode={currentMode}
      />

      {/* Welcome screen - only when no messages */}
      {messages.length === 0 && !showHelp && (
        <WelcomeScreen
          sessionId={sessionId}
          mode={currentMode}
          showHelp={true}
        />
      )}

      {/* Help panel - toggle with /help */}
      {showHelp && <HelpPanel simple={simple} />}

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
        <StreamingBubble
          content={currentResponse}
          orchestrated={!simple}
        />
      )}

      {/* Status indicator - transient, only while processing */}
      {isProcessing && statusInfo && <StatusIndicator status={statusInfo} />}

      {/* Voting progress bar - only during k-voting */}
      {isProcessing && statusInfo?.phase === "voting" && statusInfo.details?.voting_k && (
        <VotingProgress
          totalVotes={statusInfo.details.voting_k}
          votesCast={statusInfo.details.votes_cast || 0}
          consensus={statusInfo.details.votes_cast ? statusInfo.details.votes_cast - (statusInfo.details.red_flagged || 0) : 0}
          flagged={statusInfo.details.red_flagged || 0}
        />
      )}

      {/* Thinking indicator */}
      {thinkingStatus && (
        <ThinkingIndicator status={thinkingStatus} variant="melon" />
      )}

      {/* Error display */}
      {error && (
        <Panel status="error" padding={1}>
          <Text color="red">{error}</Text>
        </Panel>
      )}

      {/* Separator */}
      <Divider />

      {/* Confirmation prompt */}
      {pendingConfirmation && (
        <ConfirmPrompt
          confirmId={pendingConfirmation.confirmId}
          tool={pendingConfirmation.tool}
          args={pendingConfirmation.args}
          message={pendingConfirmation.message}
          onConfirm={handleConfirm}
        />
      )}

      {/* Input prompt */}
      <InputPrompt
        prompt={simple ? "> " : ">> "}
        onSubmit={handleSubmit}
        onExit={handleExit}
        onInterrupt={handleInterrupt}
        processing={isProcessing}
        history={history}
        promptColor={simple ? "dim" : "cyan"}
        showBorder={true}
        mode={currentMode}
      />

      {/* Footer hints */}
      <Box marginTop={1}>
        <Text color="dim">
          {isProcessing ? "Esc to interrupt · " : ""}/help · /clear · /new · /simple · Ctrl+C to exit
        </Text>
      </Box>
    </Box>
  );
};

// Re-export Message type for external use
export type { Message };

export default ChatView;
