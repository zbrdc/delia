/**
 * Chat command for interactive REPL sessions.
 */

import React from "react";
import { render } from "ink";
import { ChatView } from "../components/ChatView.js";

export interface ChatCommandOptions {
  model?: string;
  backend?: string;
  session?: string;
  api?: string;
}

export async function runChat(options: ChatCommandOptions): Promise<void> {
  const { waitUntilExit } = render(
    <ChatView
      apiUrl={options.api || "http://localhost:8201"}
      model={options.model}
      backendType={options.backend}
      sessionId={options.session}
    />
  );

  await waitUntilExit();
}
