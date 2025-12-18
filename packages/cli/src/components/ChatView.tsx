/**
 * ChatView component for interactive chat sessions.
 *
 * Provides a high-end dashboard-style interface.
 * Left: Clean Chat History (Markdown)
 * Right: System Dashboard (Model, Backend, Tokens, Tool Log)
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useApp, useStdout } from "ink";
import Spinner from "ink-spinner";
import { DeliaClient, type ChatEvent } from "../services/api.js";
import { MarkdownText } from "./MarkdownText.js";
import { InputPrompt } from "./InputPrompt.js";
import { StatusIndicator, VotingProgress, type StatusInfo } from "./StatusIndicator.js";
import { Header, WelcomeScreen } from "./Header.js";
import { MessageBubble, StreamingBubble, type Message } from "./MessageBubble.js";
import { Panel, Divider, Chip } from "./Panel.js";
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
 * Dashboard Sidebar component.
 */
const DashboardSidebar: React.FC<{
  model?: string;
  backend?: string;
  tokens: number;
  totalTokens: number;
  elapsedMs?: number;
  tools: string[];
  quality?: number;
  height: number;
}> = ({ model, backend, tokens, totalTokens, elapsedMs, tools, quality, height }) => (
  <Box flexDirection="column" width={35} marginLeft={2} height={height}>
    <Panel title="INTELLIGENCE" titleColor="magenta" borderStyle="round" borderColor="magenta">
      <Box flexDirection="column">
        <Chip label="Model" value={model || "auto"} color="cyan" />
        <Chip label="Backend" value={backend || "local"} color="yellow" />
        <Box marginTop={1}>
          <Text color="dim" bold>Performance:</Text>
        </Box>
        <Chip label="Latency" value={elapsedMs ? `${elapsedMs}ms` : "-"} color="green" />
        <Chip label="Current" value={`${tokens} tok`} color="blue" />
        <Chip label="Session" value={`${totalTokens} tok`} color="blue" />
        
        {quality !== undefined && (
          <Box marginTop={1}>
            <Text color="dim" bold>Quality Score:</Text>
            <Box marginTop={0}>
              <Text color={quality > 0.8 ? "green" : "yellow"}>
                {"█".repeat(Math.round(quality * 10))}
                <Text color="dim">{"░".repeat(10 - Math.round(quality * 10))}</Text>
                {` ${Math.round(quality * 100)}%`}
              </Text>
            </Box>
          </Box>
        )}
      </Box>
    </Panel>

    {/* TOOL LOG with fixed height to prevent pushing other elements */}
    <Box height={12}>
      <Panel title="TOOL LOG" titleColor="cyan" borderStyle="single" borderColor="dim" padding={0}>
        <Box flexDirection="column" paddingX={1}>
          {tools.length === 0 ? (
            <Text color="dim" italic>No tools used yet</Text>
          ) : (
            tools.slice(-8).map((tool, i) => (
              <Text key={i} color="cyan" wrap="truncate-end">
                <Text color="dim">› </Text>{tool}
              </Text>
            ))
          )}
        </Box>
      </Panel>
    </Box>
    
    <Box marginTop="auto" borderStyle="round" borderColor="dim" paddingX={1}>
      <Text color="magenta" bold> 🍈 DELIA </Text>
      <Text color="dim">v1.0.0</Text>
    </Box>
  </Box>
);

export const ChatView: React.FC<ChatViewProps> = ({
  apiUrl,
  model: initialModel,
  backendType,
  sessionId: initialSessionId,
  simple: initialSimple = false,
  legacyOrchestrated: initialLegacyOrchestrated = false,
  includeFileTools = true,
  workspace,
  onExit,
}) => {
  const { exit } = useApp();
  const { stdout } = useStdout();
  
  // Terminal size detection
  const [termWidth, setTermWidth] = useState(stdout?.columns || 100);
  const [termHeight, setTermHeight] = useState(stdout?.rows || 30);
  
  useEffect(() => {
    const handleResize = () => {
      setTermWidth(stdout?.columns || 100);
      setTermHeight(stdout?.rows || 30);
    };
    stdout?.on("resize", handleResize);
    return () => { stdout?.off("resize", handleResize); };
  }, [stdout]);

  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId);
  const [isProcessing, setIsProcessing] = useState(false);
  const [thinkingStatus, setThinkingStatus] = useState<string | null>(null);
  const [statusInfo, setStatusInfo] = useState<StatusInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string>("");
  const [simple, setSimple] = useState(initialSimple);
  const [legacyOrchestrated, setLegacyOrchestrated] = useState(initialLegacyOrchestrated);
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "connecting" | "disconnected" | "error">("connecting");

  // Dashboard State
  const [activeModel, setActiveModel] = useState<string | undefined>(initialModel);
  const [activeBackend, setActiveBackend] = useState<string | undefined>();
  const [lastTokens, setLastTokens] = useState(0);
  const [totalTokens, setTotalTokens] = useState(0);
  const [lastElapsedMs, setLastElapsedMs] = useState<number | undefined>();
  const [toolLog, setToolLog] = useState<string[]>([]);
  const [qualityScore, setQualityScore] = useState<number | undefined>();

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
    if (!input.trim()) return;

    // Handle commands
    if (input.startsWith("/")) {
      const cmd = input.toLowerCase();
      if (cmd === "/clear") {
        setMessages([]);
        setToolLog([]);
        setError(null);
        return;
      }
      if (cmd === "/new") {
        setMessages([]);
        setSessionId(undefined);
        setToolLog([]);
        setTotalTokens(0);
        setError(null);
        return;
      }
      if (cmd === "/simple") {
        setSimple((prev) => !prev);
        setLegacyOrchestrated(false);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: !simple 
              ? "📝 Simple mode enabled - No orchestration"
              : "✨ NLP Orchestration enabled (Default)",
            timestamp: new Date(),
          },
        ]);
        return;
      }
      if (cmd === "/agent" || cmd === "/tool") {
        setLegacyOrchestrated(true);
        setSimple(false);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: "🔧 Tool Orchestration Mode enabled - Model drives tools directly",
            timestamp: new Date(),
          },
        ]);
        return;
      }
      if (cmd === "/nlp") {
        setLegacyOrchestrated(false);
        setSimple(false);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content: "✨ NLP Orchestration Mode enabled - Delia drives intent",
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

    let responseContent = "";
    let assistantModel: string | undefined;
    let elapsedMs: number | undefined;
    let tokens: number | undefined;
    let currentTools: string[] = [];
    let hasError = false;

    try {
      await client.sendChat(
        {
          message: input,
          sessionId,
          model: initialModel,
          backendType,
          simple,
          orchestrated: legacyOrchestrated,
          includeFileTools,
          workspace,
        },
        (event: ChatEvent) => {
          switch (event.type) {
            case "session":
              if (event.created) setSessionId(event.id);
              break;

            case "thinking":
              setThinkingStatus(event.status);
              break;

            case "status":
              setStatusInfo({
                phase: event.phase as StatusInfo["phase"],
                message: event.message,
                details: event.details,
              });
              if (event.details?.model) setActiveModel(event.details.model);
              if (event.details?.backend) setActiveBackend(event.details.backend);
              if (event.details?.quality_score !== undefined) setQualityScore(event.details.quality_score);
              break;

            case "tool_call":
              const toolName = (event as any).name || "tool";
              setToolLog(prev => [...prev, toolName]);
              currentTools.push(toolName);
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
              
              if (assistantModel) setActiveModel(assistantModel);
              if (elapsedMs) setLastElapsedMs(elapsedMs);
              if (tokens) {
                setLastTokens(tokens);
                setTotalTokens(prev => prev + tokens);
              }
              break;
          }
        }
      );

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
            toolsUsed: currentTools.length > 0 ? currentTools : undefined,
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
  }, [client, sessionId, initialModel, backendType, simple, legacyOrchestrated, includeFileTools, workspace]);

  const handleExit = useCallback(() => {
    onExit?.();
    exit();
  }, [exit, onExit]);

  const handleInterrupt = useCallback(() => {
    setIsProcessing(false);
    setThinkingStatus(null);
    setStatusInfo(null);
    setCurrentResponse("");
  }, []);

  // Limit history items strictly to keep UI stable and avoid scrolling issues
  // We only show the last 4 messages + current streaming response
  const displayMessages = messages.slice(-4);
  const currentMode = simple ? "simple" : "nlp";
  
  // Calculate message area height
  const messageAreaHeight = termHeight - 10;

  return (
    <Box flexDirection="column" height={termHeight - 2} padding={1}>
      {/* Top Header */}
      <Header
        variant="minimal"
        sessionId={sessionId}
        status={connectionStatus}
        mode={currentMode}
        model={activeModel}
        backend={activeBackend}
      />

      <Box flexGrow={1} flexDirection="row">
        {/* Main Content Area */}
        <Box flexDirection="column" flexGrow={1}>
          <Box flexDirection="column" height={messageAreaHeight} overflow="hidden" paddingRight={1}>
            <Box flexDirection="column" justifyContent="flex-start">
              {messages.length === 0 && !currentResponse && (
                <WelcomeScreen sessionId={sessionId} mode={currentMode} showHelp={false} />
              )}

              {displayMessages.map((msg, i) => (
                <MessageBubble
                  key={i}
                  message={msg}
                  isLast={i === displayMessages.length - 1}
                />
              ))}

              {currentResponse && (
                <Box borderStyle="single" borderColor="magenta" paddingX={1} marginY={1}>
                  <StreamingBubble content={currentResponse} orchestrated={!simple} />
                </Box>
              )}

              {isProcessing && statusInfo && (
                <Box marginLeft={2}>
                  <StatusIndicator status={statusInfo} />
                </Box>
              )}
              
              {thinkingStatus && (
                <Box marginLeft={2}>
                  <ThinkingIndicator status={thinkingStatus} variant="melon" />
                </Box>
              )}

              {error && (
                <Panel status="error" padding={1}>
                  <Text color="red">{error}</Text>
                </Panel>
              )}
            </Box>
          </Box>

          <Divider />

          {/* Input Area */}
          <Box height={3} flexShrink={0}>
            <InputPrompt
              prompt={simple ? "  > " : " ✨ >> "}
              onSubmit={handleSubmit}
              onExit={handleExit}
              onInterrupt={handleInterrupt}
              processing={isProcessing}
              history={history}
              promptColor="cyan"
              showBorder={false}
              mode={currentMode}
            />
          </Box>
        </Box>

        {/* System Dashboard Sidebar */}
        {termWidth > 80 && (
          <DashboardSidebar
            model={activeModel}
            backend={activeBackend}
            tokens={lastTokens}
            totalTokens={totalTokens}
            elapsedMs={lastElapsedMs}
            tools={toolLog}
            quality={qualityScore}
            height={messageAreaHeight + 4}
          />
        )}
      </Box>

      {/* Shortcut hints */}
      <Box height={1} marginTop={0}>
        <Text color="dim">
          {isProcessing ? "Esc interrupt · " : ""}/help · /clear · /new · /nlp · /simple · Ctrl+C quit
        </Text>
      </Box>
    </Box>
  );
};

export type { Message };
export default ChatView;