/**
 * Agent command - run autonomous agent tasks.
 *
 * Usage: delia-cli agent "task description"
 */

import React, { useState, useEffect } from "react";
import { render } from "ink";
import {
  DeliaClient,
  type AgentEvent,
  type AgentRunOptions,
  type StatusEvent,
} from "../lib/api.js";
import {
  StreamingResponse,
  type ToolCall,
  type StatusInfo,
} from "../components/StreamingResponse.js";

export interface AgentCommandOptions {
  model?: string;
  workspace?: string;
  maxIterations?: number;
  tools?: string;
  backend?: string;
  apiUrl?: string;
  // Permission flags
  allowWrite?: boolean;
  allowExec?: boolean;
  yolo?: boolean;
}

interface AgentState {
  thinking: string | null;
  statusInfo: StatusInfo | null;
  response: string;
  toolCalls: ToolCall[];
  done: boolean;
  success: boolean;
  model?: string;
  backend?: string;
  elapsed_ms?: number;
  iterations?: number;
}

const AgentApp: React.FC<{
  task: string;
  options: AgentCommandOptions;
}> = ({ task, options }) => {
  const [state, setState] = useState<AgentState>({
    thinking: "Connecting...",
    statusInfo: null,
    response: "",
    toolCalls: [],
    done: false,
    success: false,
  });

  useEffect(() => {
    const client = new DeliaClient(options.apiUrl);

    const runOptions: AgentRunOptions = {
      task,
      model: options.model,
      workspace: options.workspace,
      maxIterations: options.maxIterations ?? 10,
      tools: options.tools?.split(",").map((t) => t.trim()),
      backendType: options.backend,
      // Permission flags
      allowWrite: options.allowWrite ?? false,
      allowExec: options.allowExec ?? false,
      yolo: options.yolo ?? false,
    };

    const handleEvent = (event: AgentEvent) => {
      switch (event.type) {
        case "thinking":
          setState((s) => ({ ...s, thinking: event.status }));
          break;

        case "status":
          setState((s) => ({
            ...s,
            statusInfo: {
              phase: event.phase,
              message: event.message,
              details: event.details,
            },
          }));
          break;

        case "tool_call":
          setState((s) => ({
            ...s,
            toolCalls: [
              ...s.toolCalls,
              {
                name: event.name,
                args: event.args,
                status: "running",
              },
            ],
          }));
          break;

        case "tool_result":
          setState((s) => ({
            ...s,
            toolCalls: s.toolCalls.map((tc, i) =>
              i === s.toolCalls.length - 1
                ? {
                    ...tc,
                    status: event.success ? "success" : "error",
                    output: event.output,
                    elapsed_ms: event.elapsed_ms,
                  }
                : tc
            ),
          }));
          break;

        case "token":
          setState((s) => ({
            ...s,
            response: s.response + event.content,
          }));
          break;

        case "response":
          setState((s) => ({
            ...s,
            response: event.content,
            thinking: null,
          }));
          break;

        case "error":
          setState((s) => ({
            ...s,
            thinking: null,
            response: `Error: ${event.message}`,
            done: true,
            success: false,
          }));
          break;

        case "done":
          setState((s) => ({
            ...s,
            thinking: null,
            statusInfo: null,
            done: true,
            success: event.success,
            model: event.model,
            backend: event.backend,
            elapsed_ms: event.elapsed_ms,
            iterations: event.iterations,
          }));
          break;
      }
    };

    client.runAgent(runOptions, handleEvent).catch((err) => {
      setState((s) => ({
        ...s,
        thinking: null,
        response: `Connection error: ${err.message}`,
        done: true,
        success: false,
      }));
    });
  }, [task, options]);

  return (
    <StreamingResponse
      task={task}
      thinking={state.thinking}
      statusInfo={state.statusInfo}
      response={state.response}
      toolCalls={state.toolCalls}
      done={state.done}
      success={state.success}
      model={state.model}
      backend={state.backend}
      elapsed_ms={state.elapsed_ms}
      iterations={state.iterations}
    />
  );
};

export async function runAgent(
  task: string,
  options: AgentCommandOptions
): Promise<void> {
  const { waitUntilExit } = render(
    <AgentApp task={task} options={options} />
  );
  await waitUntilExit();
}
