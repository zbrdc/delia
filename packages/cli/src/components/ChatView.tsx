/**
 * ChatView component - Minimal Document Interface
 */

import React, { useState, useCallback, useEffect } from "react";
import { Box, Text, useApp, useStdout } from "ink";
import { DeliaClient, type ChatEvent } from "../services/api.js";
import { InputPrompt } from "./InputPrompt.js";
import { MessageBubble, StreamingBubble, type Message } from "./MessageBubble.js";

export interface ChatViewProps {
  apiUrl: string;
  model?: string;
  backendType?: string;
  sessionId?: string;
  simple?: boolean;
  legacyOrchestrated?: boolean;
  includeFileTools?: boolean;
  workspace?: string;
  onExit?: () => void;
}

export const ChatView: React.FC<ChatViewProps> = ({
  apiUrl,
  model: initialModel,
  backendType,
  sessionId: initialSessionId,
  simple = false,
  legacyOrchestrated = false,
  includeFileTools = true,
  workspace,
  onExit,
}) => {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [termHeight, setTermHeight] = useState(stdout?.rows || 30);
  
  useEffect(() => {
    const handleResize = () => setTermHeight(stdout?.rows || 30);
    stdout?.on("resize", handleResize);
    return () => { stdout?.off("resize", handleResize); };
  }, [stdout]);

  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId);
  const [isProcessing, setIsProcessing] = useState(false);
  const [thinkingStatus, setThinkingStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string>("");

  const client = new DeliaClient(apiUrl);

  useEffect(() => {
    client.health().catch(() => {});
  }, [apiUrl]);

  const handleSubmit = useCallback(async (input: string) => {
    if (!input.trim()) return;

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

    let responseContent = "";
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
          },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsProcessing(false);
      setThinkingStatus(null);
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
    setCurrentResponse("");
  }, []);

  const displayMessages = messages.slice(-10);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box flexDirection="column" flexGrow={1} overflow="hidden">
        <Box flexDirection="column" justifyContent="flex-end">
          {displayMessages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {currentResponse && (
            <Box marginY={1}>
              <StreamingBubble content={currentResponse} />
            </Box>
          )}

          {thinkingStatus && !currentResponse && (
             <Box marginY={1}>
               <Text color="dim">⠋ {thinkingStatus}</Text>
             </Box>
          )}

          {error && (
            <Box marginY={1}>
              <Text color="red">Error: {error}</Text>
            </Box>
          )}
        </Box>
      </Box>

      {/* Input Area - Flexible Height */}
      <Box marginY={1}>
        <InputPrompt
          prompt="> "
          onSubmit={handleSubmit}
          onExit={handleExit}
          onInterrupt={handleInterrupt}
          processing={isProcessing}
          history={history}
          promptColor="magenta"
        />
      </Box>
    </Box>
  );
};

export type { Message };
export default ChatView;