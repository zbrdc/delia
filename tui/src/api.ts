/**
 * Delia API Client - Connects to Delia HTTP API server (delia api)
 *
 * Endpoints:
 *   POST /api/chat     - SSE streaming chat with orchestration
 *   GET  /api/health   - Health check
 *   GET  /api/status   - Full system status
 *   GET  /api/models   - Available models per tier
 */

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface StreamEvent {
  type: 'session' | 'intent' | 'status' | 'thinking' | 'frustration' |
        'token' | 'response' | 'error' | 'done' | 'tool_call' | 'tool_result' |
        'message' | 'orchestration' | 'quality' | 'confirm';
  data: Record<string, unknown>;
}

export interface ConfirmRequest {
  confirm_id: string;
  confirmed: boolean;
  allow_all?: boolean;
}

export interface ChatOptions {
  model?: string;
  backendType?: string;
  simple?: boolean;  // Skip orchestration
  allowWrite?: boolean;  // Enable file write operations
  allowExec?: boolean;   // Enable shell execution
  yolo?: boolean;        // Skip all security prompts
}

export interface HealthResponse {
  status: string;
  backends: Array<{
    id: string;
    name: string;
    provider: string;
    type: string;
  }>;
}

export class DeliaClient {
  private baseUrl: string;
  private sessionId: string | null = null;

  constructor(baseUrl = 'http://localhost:34589') {
    this.baseUrl = baseUrl;
  }

  async health(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
  }

  async status(): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/api/status`);
    if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
    return res.json();
  }

  async models(): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/api/models`);
    if (!res.ok) throw new Error(`Models check failed: ${res.status}`);
    return res.json();
  }

  /**
   * Confirm or deny a dangerous tool execution.
   * Called when a 'confirm' event is received during chat.
   */
  async confirm(request: ConfirmRequest): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/api/agent/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      throw new Error(`Confirm failed: ${res.status}`);
    }
    const data = await res.json();
    return data.confirmed === true;
  }

  /**
   * Stream chat response via SSE.
   * Uses the /api/chat endpoint with NLP orchestration by default.
   */
  async *chat(message: string, options: ChatOptions = {}): AsyncGenerator<StreamEvent> {
    const body = {
      message,
      session_id: this.sessionId,
      model: options.model,
      backend_type: options.backendType,
      simple: options.simple || false,
      allow_write: options.allowWrite || false,
      allow_exec: options.allowExec || false,
      yolo: options.yolo || false,
    };

    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      yield { type: 'error', data: { message: `HTTP ${res.status}: ${res.statusText}` } };
      return;
    }

    // Parse SSE stream
    const reader = res.body?.getReader();
    if (!reader) {
      yield { type: 'error', data: { message: 'No response body' } };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let currentEventType = 'message';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          // Capture event type for next data line
          currentEventType = line.slice(7).trim();
          continue;
        }
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            // Capture session ID from session events
            if (data.id && data.created) {
              this.sessionId = data.id;
            }

            yield { type: currentEventType as StreamEvent['type'], data };

            // Reset event type after yielding
            currentEventType = 'message';
          } catch {
            // Ignore parse errors
          }
        }
      }
    }
  }

  get currentSessionId(): string | null {
    return this.sessionId;
  }

  clearSession(): void {
    this.sessionId = null;
  }

  /**
   * Compact the current session's conversation history.
   * Reduces token count while preserving key information.
   */
  async compact(force = false): Promise<CompactResponse> {
    if (!this.sessionId) {
      return { success: false, error: 'No active session' };
    }
    const url = `${this.baseUrl}/api/sessions/${this.sessionId}/compact${force ? '?force=true' : ''}`;
    const res = await fetch(url, { method: 'POST' });
    return res.json();
  }

  /**
   * Get compaction stats for the current session.
   */
  async sessionStats(): Promise<StatsResponse> {
    if (!this.sessionId) {
      return { session_id: '', total_messages: 0, total_tokens: 0, needs_compaction: false };
    }
    const res = await fetch(`${this.baseUrl}/api/sessions/${this.sessionId}/stats`);
    return res.json();
  }
}

export interface CompactResponse {
  success: boolean;
  error?: string;
  session_id?: string;
  messages_compacted?: number;
  tokens_saved?: number;
  compression_ratio?: number;
  summary_preview?: string;
}

export interface StatsResponse {
  session_id: string;
  total_messages: number;
  total_tokens: number;
  needs_compaction: boolean;
  threshold_tokens?: number;
  compactable_messages?: number;
}

export const client = new DeliaClient();
