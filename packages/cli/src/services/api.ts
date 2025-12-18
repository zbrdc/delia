/**
 * Delia API Client
 *
 * Handles HTTP requests and SSE streaming communication with the Python backend.
 */

import { createParser } from "eventsource-parser";

/**
 * Event types received from the backend via SSE.
 */
export type ChatEvent =
  | { type: "session"; id: string; created: boolean }
  | { type: "status"; phase: string; message: string; details?: any }
  | { type: "thinking"; status: string }
  | { type: "token"; content: string }
  | { type: "response"; content: string }
  | { type: "error"; message: string }
  | { type: "confirm"; confirm_id: string; tool: string; args: any; message: string }
  | { type: "tools"; calls: any[]; count: number }
  | { type: "tool_call"; iteration: number; calls: any[] }
  | { type: "tool_result"; name: string; success: boolean; output_preview: string; elapsed_ms?: number; output?: string }
  | { type: "done"; model: string; elapsed_ms: number; tokens: number; tools_used?: string[]; orchestrated?: boolean; success?: boolean; iterations?: number };

export type AgentEvent = ChatEvent;
export type StatusEvent = { type: "status"; phase: string; message: string; details?: any };
export type ConfirmEvent = { type: "confirm"; confirm_id: string; tool: string; args: any; message: string };

/**
 * Chat request options.
 */
export interface ChatOptions {
  message: string;
  sessionId?: string;
  model?: string;
  backendType?: string;
  simple?: boolean;
  orchestrated?: boolean;
  includeFileTools?: boolean;
  workspace?: string;
}

/**
 * Agent run options.
 */
export interface AgentRunOptions {
  task: string;
  model?: string;
  workspace?: string;
  maxIterations?: number;
  tools?: string[];
  backendType?: string;
  allowWrite?: boolean;
  allowExec?: boolean;
  yolo?: boolean;
}

export interface Callbacks {
  onEvent: (event: any) => void;
}

/**
 * Delia API Client implementation.
 */
export class DeliaClient {
  private baseUrl: string;

  constructor(apiUrl?: string) {
    // Default to the standard Delia port if not specified
    this.baseUrl = apiUrl || "http://localhost:34589";
    // Remove trailing slash if present
    if (this.baseUrl.endsWith("/")) {
      this.baseUrl = this.baseUrl.slice(0, -1);
    }
  }

  /**
   * Check backend health.
   */
  async health(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Send a chat message and stream the response via SSE.
   */
  async sendChat(options: ChatOptions, callbacks: Callbacks | ((event: ChatEvent) => void)): Promise<void> {
    // Handle both old function-style and new object-style callbacks
    const onEvent = typeof callbacks === "function" ? callbacks : callbacks.onEvent;
    
    if (typeof callbacks !== "function" && !callbacks.onEvent) {
        throw new Error("callbacks.onEvent is required");
    }

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: options.message,
        session_id: options.sessionId,
        model: options.model,
        backend_type: options.backendType,
        simple: options.simple,
        orchestrated: options.orchestrated,
        include_file_tools: options.includeFileTools,
        workspace: options.workspace,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `Request failed: ${response.status}`);
    }

    await this.processStream(response, onEvent);
  }

  /**
   * Run an autonomous agent task and stream events.
   */
  async runAgent(options: AgentRunOptions, callbacks: Callbacks | ((event: AgentEvent) => void)): Promise<void> {
    const onEvent = typeof callbacks === "function" ? callbacks : callbacks.onEvent;

    const response = await fetch(`${this.baseUrl}/api/agent/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        task: options.task,
        model: options.model,
        workspace: options.workspace,
        max_iterations: options.maxIterations,
        tools: options.tools,
        backend_type: options.backendType,
        allow_write: options.allowWrite,
        allow_exec: options.allowExec,
        yolo: options.yolo,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `Agent execution failed: ${response.status}`);
    }

    await this.processStream(response, onEvent);
  }

  /**
   * Confirm or deny a dangerous tool execution.
   */
  async confirmTool(params: { confirmId: string; confirmed: boolean; allowAll?: boolean }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/agent/confirm`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        confirm_id: params.confirmId,
        confirmed: params.confirmed,
        allow_all: params.allowAll,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `Confirmation failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Process an SSE stream from the backend.
   */
  private async processStream(response: Response, onEvent: (event: ChatEvent) => void): Promise<void> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body is null");
    }

    const parser = createParser({
      onEvent: (event) => {
        if (event.event) {
          try {
            const data = JSON.parse(event.data);
            onEvent({ type: event.event as any, ...data });
          } catch (e) {
            console.error("Failed to parse SSE data:", e);
          }
        }
      }
    });

    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
    }
  }
}
