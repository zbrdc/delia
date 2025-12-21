import React, { useState, useEffect, useCallback } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Spinner from 'ink-spinner';
import TextInput from 'ink-text-input';
import { DeliaClient, StreamEvent } from './api.js';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  timestamp: Date;
}

interface PendingConfirmation {
  confirm_id: string;
  tool: string;
  args: Record<string, unknown>;
  message: string;
}

type Status = 'ready' | 'connecting' | 'thinking' | 'streaming' | 'error' | 'confirming';

interface AppProps {
  serverUrl: string;
  allowWrite?: boolean;
  allowExec?: boolean;
  yolo?: boolean;
}

export function App({ serverUrl, allowWrite = true, allowExec = true, yolo = false }: AppProps) {
  const { exit } = useApp();
  const [client] = useState(() => new DeliaClient(serverUrl));
  const [status, setStatus] = useState<Status>('connecting');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamBuffer, setStreamBuffer] = useState('');
  const [backendName, setBackendName] = useState('connecting...');
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);

  // Check health on mount
  useEffect(() => {
    client.health()
      .then(h => {
        if (h.backends.length > 0) {
          setBackendName(h.backends[0].name || 'local');
        }
        setStatus('ready');
      })
      .catch(() => {
        setBackendName('offline');
        setError('Cannot connect to Delia API. Start it with: delia api');
        setStatus('error');
      });
  }, [client]);

  // Handle keyboard shortcuts
  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
    }
  });

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || status !== 'ready') return;

    // Handle exit commands
    if (trimmed.toLowerCase() === 'exit' || trimmed.toLowerCase() === 'quit') {
      exit();
      return;
    }

    // Handle clear command
    if (trimmed.toLowerCase() === 'clear' || trimmed.toLowerCase() === '/clear') {
      setMessages([]);
      setInput('');
      client.clearSession();
      return;
    }

    // Handle help command
    if (trimmed.toLowerCase() === '/help' || trimmed.toLowerCase() === 'help') {
      setInput('');
      setMessages(prev => [...prev, {
        role: 'assistant' as const,
        content: `**Available Commands:**

/help     - Show this help message
/clear    - Clear chat history and start fresh
/compact  - Summarize chat history to reduce context size
/stats    - Show session statistics (tokens, messages)
exit      - Exit the chat

**Security:**
File and shell operations are enabled by default. Delia will prompt you for approval before running dangerous operations. Respond with:
- **y** (yes) - Allow this operation
- **n** (no) - Deny this operation
- **a** (all) - Allow all future operations in this session`,
        timestamp: new Date(),
      }]);
      return;
    }

    // Handle compact command
    if (trimmed.toLowerCase() === '/compact') {
      setInput('');
      setStatus('thinking');
      try {
        const result = await client.compact(true);
        if (result.success) {
          setMessages(prev => [...prev, {
            role: 'assistant' as const,
            content: `✅ **Session Compacted**

Messages summarized: ${result.messages_compacted}
Tokens saved: ${result.tokens_saved?.toLocaleString()}
Compression: ${((result.compression_ratio || 0) * 100).toFixed(0)}%

${result.summary_preview ? `Preview: ${result.summary_preview}...` : ''}`,
            timestamp: new Date(),
          }]);
        } else {
          setError(result.error || 'Compaction failed');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Compaction failed');
      }
      setStatus('ready');
      return;
    }

    // Handle stats command
    if (trimmed.toLowerCase() === '/stats') {
      setInput('');
      setStatus('thinking');
      try {
        const stats = await client.sessionStats();
        setMessages(prev => [...prev, {
          role: 'assistant' as const,
          content: `**Session Stats**

Messages: ${stats.total_messages}
Tokens: ${stats.total_tokens.toLocaleString()}
Needs compaction: ${stats.needs_compaction ? 'Yes ⚠️' : 'No ✓'}
${stats.threshold_tokens ? `Threshold: ${stats.threshold_tokens.toLocaleString()} tokens` : ''}`,
          timestamp: new Date(),
        }]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to get stats');
      }
      setStatus('ready');
      return;
    }

    setInput('');
    setError(null);

    // Add user message
    const userMsg: ChatMessage = { role: 'user', content: trimmed, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);

    setStatus('thinking');
    setStreamBuffer('');
    setCurrentModel(null);

    try {
      let fullResponse = '';
      let responseModel = '';

      for await (const event of client.chat(trimmed, { allowWrite, allowExec, yolo })) {
        const data = event.data;

        switch (event.type) {
          case 'session':
            // Session created/restored
            break;
          case 'intent':
            // Intent detected
            break;
          case 'status':
            // Update model display
            if (data.details && typeof data.details === 'object') {
              const details = data.details as Record<string, unknown>;
              if (details.model) {
                responseModel = String(details.model);
                setCurrentModel(responseModel);
              }
            }
            break;
          case 'thinking':
            setStatus('thinking');
            break;
          case 'token':
            if (data.content) {
              fullResponse += String(data.content);
              setStreamBuffer(fullResponse);
              setStatus('streaming');
            }
            break;
          case 'response':
            fullResponse = String(data.content || data.message || fullResponse);
            break;
          case 'error':
            setError(String(data.message || 'Unknown error'));
            setStatus('error');
            return;
          case 'done':
            if (data.model) {
              responseModel = String(data.model);
            }
            break;
          case 'confirm':
            // Dangerous operation needs user approval
            setPendingConfirmation({
              confirm_id: String(data.confirm_id),
              tool: String(data.tool),
              args: data.args as Record<string, unknown>,
              message: String(data.message || `Allow ${data.tool}?`),
            });
            setStatus('confirming');
            break;
        }
      }

      // Add assistant message
      if (fullResponse) {
        const assistantMsg: ChatMessage = {
          role: 'assistant',
          content: fullResponse,
          model: responseModel || undefined,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMsg]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed');
      setStatus('error');
      return;
    }

    setStreamBuffer('');
    setStatus('ready');
    setCurrentModel(null);
  }, [input, status, client, exit]);

  // Handle confirmation response (y/n/a)
  const handleConfirmation = useCallback(async (response: string) => {
    if (!pendingConfirmation) return;

    const lower = response.trim().toLowerCase();
    let confirmed = false;
    let allowAll = false;

    if (lower === 'y' || lower === 'yes') {
      confirmed = true;
    } else if (lower === 'a' || lower === 'all' || lower === 'allow all') {
      confirmed = true;
      allowAll = true;
    } else if (lower === 'n' || lower === 'no') {
      confirmed = false;
    } else {
      // Invalid response, show hint
      setError('Type y (yes), n (no), or a (allow all)');
      return;
    }

    try {
      await client.confirm({
        confirm_id: pendingConfirmation.confirm_id,
        confirmed,
        allow_all: allowAll,
      });

      // Add a message showing the decision
      const decisionMsg = confirmed
        ? (allowAll ? 'Allowed all future operations' : `Allowed: ${pendingConfirmation.tool}`)
        : `Denied: ${pendingConfirmation.tool}`;
      setMessages(prev => [...prev, {
        role: 'user' as const,
        content: decisionMsg,
        timestamp: new Date(),
      }]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Confirmation failed');
    }

    // Clear confirmation state - the stream will continue
    setPendingConfirmation(null);
    setInput('');
    setStatus('thinking');
    setError(null);
  }, [pendingConfirmation, client]);

  return (
    <Box flexDirection="column" height="100%">
      {/* Header */}
      <Box borderStyle="single" borderColor="cyan" paddingX={1}>
        <Text color="cyan" bold>DELIA</Text>
        <Text> │ </Text>
        <Text color="gray">{backendName}</Text>
        <Text> │ </Text>
        <StatusIndicator status={status} model={currentModel} />
      </Box>

      {/* Chat History */}
      <Box flexDirection="column" flexGrow={1} paddingX={1} marginY={1}>
        {messages.length === 0 && (
          <Text color="gray" italic>Type a message to start chatting...</Text>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {/* Streaming buffer */}
        {streamBuffer && (
          <Box flexDirection="column" marginTop={1}>
            <Text wrap="wrap">{streamBuffer}<Text color="gray">▌</Text></Text>
          </Box>
        )}
        {/* Confirmation prompt */}
        {pendingConfirmation && (
          <Box flexDirection="column" marginTop={1} borderStyle="round" borderColor="yellow" paddingX={1}>
            <Text color="yellow" bold>Security Approval Required</Text>
            <Box marginTop={1}>
              <Text>Tool: </Text>
              <Text color="cyan" bold>{pendingConfirmation.tool}</Text>
            </Box>
            <Box>
              <Text>Args: </Text>
              <Text color="gray">{JSON.stringify(pendingConfirmation.args, null, 0).slice(0, 60)}...</Text>
            </Box>
            <Box marginTop={1}>
              <Text color="yellow">[y]es  [n]o  [a]llow all</Text>
            </Box>
          </Box>
        )}
        {/* Error display */}
        {error && (
          <Box marginTop={1}>
            <Text color="red" bold>Error: </Text>
            <Text color="red">{error}</Text>
          </Box>
        )}
      </Box>

      {/* Input */}
      <Box borderStyle="single" borderColor={status === 'confirming' ? 'yellow' : status === 'ready' ? 'green' : 'gray'} paddingX={1}>
        <Text color={status === 'confirming' ? 'yellow' : 'green'} bold>{'>'} </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={status === 'confirming' ? handleConfirmation : sendMessage}
          placeholder={status === 'confirming' ? 'y/n/a' : status === 'ready' ? 'Type your message...' : ''}
        />
      </Box>

      {/* Footer */}
      <Box paddingX={1}>
        <Text color="gray" dimColor>Ctrl+C to exit │ Enter to send</Text>
      </Box>
    </Box>
  );
}

function StatusIndicator({ status, model }: { status: Status; model?: string | null }) {
  switch (status) {
    case 'connecting':
      return (
        <Text color="yellow">
          <Spinner type="dots" /> connecting
        </Text>
      );
    case 'ready':
      return <Text color="green">● ready</Text>;
    case 'thinking':
      return (
        <Text color="yellow">
          <Spinner type="dots" /> {model ? `${model}` : 'thinking'}
        </Text>
      );
    case 'streaming':
      return (
        <Text color="cyan">
          <Spinner type="dots" /> {model || 'streaming'}
        </Text>
      );
    case 'confirming':
      return <Text color="yellow">● approval needed</Text>;
    case 'error':
      return <Text color="red">● error</Text>;
  }
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  if (isUser) {
    // User messages: simple prompt style
    return (
      <Box marginTop={1}>
        <Text color="cyan" dimColor>❯ </Text>
        <Text wrap="wrap">{message.content}</Text>
      </Box>
    );
  }

  // Assistant messages: clean response with optional model tag
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text wrap="wrap">{message.content}</Text>
      </Box>
      {message.model && (
        <Box marginTop={0}>
          <Text color="gray" dimColor>— {message.model}</Text>
        </Box>
      )}
    </Box>
  );
}

export default App;
