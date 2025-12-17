/**
 * Header component with ASCII art banner and status bar.
 *
 * Provides a visually appealing welcome screen and persistent header.
 */

import React from "react";
import { Box, Text } from "ink";

// ASCII art logo - compact version
const LOGO_SMALL = `
 ╭─────────────────────────────────╮
 │     ╺━━━━━━━━━━━━━━━━━━━━━━╸    │
 │  ██████╗ ███████╗██╗     ██╗ █████╗  │
 │  ██╔══██╗██╔════╝██║     ██║██╔══██╗ │
 │  ██║  ██║█████╗  ██║     ██║███████║ │
 │  ██║  ██║██╔══╝  ██║     ██║██╔══██║ │
 │  ██████╔╝███████╗███████╗██║██║  ██║ │
 │  ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝ │
 │     ╺━━━━━━━━━━━━━━━━━━━━━━╸    │
 ╰─────────────────────────────────╯
`;

// Minimal header for during chat
const LOGO_MINIMAL = "🍈 Delia";

export interface HeaderProps {
  /** Show full banner or minimal header */
  variant?: "full" | "minimal" | "compact";
  /** Session ID if connected */
  sessionId?: string;
  /** Connection status */
  status?: "connected" | "connecting" | "disconnected" | "error";
  /** Current mode */
  mode?: "simple" | "nlp" | "orchestrated";
  /** Model name if set */
  model?: string;
  /** Backend name */
  backend?: string;
}

const StatusDot: React.FC<{ status: HeaderProps["status"] }> = ({ status }) => {
  const colors: Record<string, string> = {
    connected: "green",
    connecting: "yellow",
    disconnected: "dim",
    error: "red",
  };
  const symbols: Record<string, string> = {
    connected: "●",
    connecting: "◐",
    disconnected: "○",
    error: "✗",
  };
  return (
    <Text color={colors[status || "disconnected"]}>
      {symbols[status || "disconnected"]}
    </Text>
  );
};

const ModeBadge: React.FC<{ mode: HeaderProps["mode"] }> = ({ mode }) => {
  const colors: Record<string, string> = {
    simple: "dim",
    nlp: "magenta",
    orchestrated: "cyan",
  };
  const labels: Record<string, string> = {
    simple: "SIMPLE",
    nlp: "NLP",
    orchestrated: "ORCH",
  };
  return (
    <Text color={colors[mode || "simple"]} bold>
      [{labels[mode || "simple"]}]
    </Text>
  );
};

export const Header: React.FC<HeaderProps> = ({
  variant = "compact",
  sessionId,
  status = "disconnected",
  mode = "nlp",
  model,
  backend,
}) => {
  if (variant === "full") {
    return (
      <Box flexDirection="column" marginBottom={1}>
        {/* Logo */}
        <Box justifyContent="center">
          <Text color="magenta" bold>
            {LOGO_SMALL}
          </Text>
        </Box>

        {/* Tagline */}
        <Box justifyContent="center">
          <Text color="dim" italic>
            Multi-Model AI Orchestration for Local LLMs
          </Text>
        </Box>

        {/* Status bar */}
        <Box
          marginTop={1}
          paddingX={2}
          paddingY={1}
          borderStyle="round"
          borderColor="dim"
          justifyContent="space-between"
        >
          <Box>
            <StatusDot status={status} />
            <Text color="dim"> {status}</Text>
            {sessionId && (
              <Text color="dim"> · {sessionId.slice(0, 8)}...</Text>
            )}
          </Box>
          <Box>
            <ModeBadge mode={mode} />
            {model && <Text color="dim"> · {model}</Text>}
            {backend && <Text color="dim"> @ {backend}</Text>}
          </Box>
        </Box>
      </Box>
    );
  }

  if (variant === "compact") {
    return (
      <Box
        paddingX={2}
        paddingY={1}
        borderStyle="round"
        borderColor="magenta"
        justifyContent="space-between"
        marginBottom={1}
      >
        {/* Left: Logo and status */}
        <Box>
          <Text color="magenta" bold>
            🍈 Delia
          </Text>
          <Text color="dim"> │ </Text>
          <StatusDot status={status} />
          <Text color="dim"> {status}</Text>
        </Box>

        {/* Right: Mode and model */}
        <Box>
          <ModeBadge mode={mode} />
          {model && (
            <>
              <Text color="dim"> · </Text>
              <Text color="cyan">{model}</Text>
            </>
          )}
        </Box>
      </Box>
    );
  }

  // Minimal variant - just inline text
  return (
    <Box marginBottom={1}>
      <Text color="magenta" bold>
        {LOGO_MINIMAL}
      </Text>
      <Text color="dim"> · </Text>
      <ModeBadge mode={mode} />
      {sessionId && <Text color="dim"> · {sessionId.slice(0, 8)}</Text>}
    </Box>
  );
};

/**
 * Welcome screen shown before first message.
 */
export interface WelcomeScreenProps {
  sessionId?: string;
  mode?: "simple" | "nlp" | "orchestrated";
  showHelp?: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  sessionId,
  mode = "nlp",
  showHelp = true,
}) => {
  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      {/* Logo box */}
      <Box
        borderStyle="double"
        borderColor="magenta"
        paddingX={4}
        paddingY={1}
        flexDirection="column"
        alignItems="center"
      >
        <Text color="magenta" bold>
          ╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸
        </Text>
        <Box marginY={1}>
          <Text color="magenta" bold>
            🍈 DELIA
          </Text>
          <Text color="dim"> - Multi-Model AI Orchestrator</Text>
        </Box>
        <Text color="magenta" bold>
          ╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸
        </Text>
      </Box>

      {/* Mode indicator */}
      <Box marginTop={1} flexDirection="column">
        {mode === "nlp" && (
          <>
            <Box>
              <Text color="magenta" bold>
                ✨ NLP Orchestration Mode
              </Text>
            </Box>
            <Box marginLeft={2}>
              <Text color="dim">
                Delia automatically detects your intent:
              </Text>
            </Box>
            <Box marginLeft={4} flexDirection="column">
              <Text color="cyan">• "verify this" → K-voting for reliability</Text>
              <Text color="cyan">• "compare models" → Multi-model comparison</Text>
              <Text color="cyan">• "think deeply" → Extended reasoning</Text>
            </Box>
          </>
        )}
        {mode === "simple" && (
          <>
            <Box>
              <Text color="dim" bold>
                📝 Simple Mode
              </Text>
            </Box>
            <Box marginLeft={2}>
              <Text color="dim">
                Direct single-model chat, no orchestration
              </Text>
            </Box>
          </>
        )}
        {mode === "orchestrated" && (
          <>
            <Box>
              <Text color="cyan" bold>
                🔧 Tool Orchestration Mode
              </Text>
            </Box>
            <Box marginLeft={2}>
              <Text color="dim">
                Multi-model tool calling with explicit control
              </Text>
            </Box>
          </>
        )}
      </Box>

      {/* Help section */}
      {showHelp && (
        <Box
          marginTop={1}
          paddingX={2}
          paddingY={1}
          borderStyle="single"
          borderColor="dim"
          flexDirection="column"
        >
          <Text color="dim" bold>
            Commands:
          </Text>
          <Box marginLeft={2} flexDirection="column">
            <Text color="dim">/help · /clear · /new · /simple · /exit</Text>
          </Box>
        </Box>
      )}

      {/* Session info */}
      {sessionId && (
        <Box marginTop={1}>
          <Text color="dim">
            Session: {sessionId.slice(0, 8)}...
          </Text>
        </Box>
      )}
    </Box>
  );
};

export default Header;
