/**
 * Chat command for interactive REPL sessions.
 * 
 * Supports orchestrated mode for multi-model tool calling.
 */

import React from "react";
import { render } from "ink";
import { ChatView } from "../components/ChatView.js";

export interface ChatCommandOptions {
  model?: string;
  backend?: string;
  session?: string;
  api?: string;
  /** Enable orchestrated mode with multi-model tools */
  orchestrated?: boolean;
  /** Include file/web tools in orchestrated mode */
  fileTools?: boolean;
  /** Workspace path for file operations */
  workspace?: string;
}

export async function runChat(options: ChatCommandOptions): Promise<void> {
  const apiUrl = options.api || process.env.DELIA_API_URL;

  if (!apiUrl) {
    console.error("Error: API URL required. Set DELIA_API_URL or use --api-url");
    console.error("  Tip: Run 'delia chat' instead to auto-start the API server");
    process.exit(1);
  }

  const { waitUntilExit } = render(
    <ChatView
      apiUrl={apiUrl}
      model={options.model}
      backendType={options.backend}
      sessionId={options.session}
      orchestrated={options.orchestrated}
      includeFileTools={options.fileTools}
      workspace={options.workspace}
    />
  );

  await waitUntilExit();
}
