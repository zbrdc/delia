/**
 * InputPrompt component for interactive text input.
 *
 * Provides a readline-style input with history support and enhanced styling.
 */

import React, { useState, useCallback } from "react";
import { Box, Text, useInput } from "ink";

export interface InputPromptProps {
  /** Prompt prefix (e.g., "> " or "you: ") */
  prompt?: string;
  /** Placeholder text when empty */
  placeholder?: string;
  /** Called when user submits input */
  onSubmit: (value: string) => void;
  /** Called when user requests exit (Ctrl+C or /exit) */
  onExit?: () => void;
  /** Called when user presses Escape (for interrupt) */
  onInterrupt?: () => void;
  /** Whether input is disabled */
  disabled?: boolean;
  /** Whether processing is in progress (shows different UI but allows input) */
  processing?: boolean;
  /** Input history for up/down navigation */
  history?: string[];
  /** Prompt color */
  promptColor?: string;
  /** Show border around input area */
  showBorder?: boolean;
  /** Mode indicator (shown as badge) */
  mode?: "simple" | "nlp" | "orchestrated";
  /** Show character count */
  showCharCount?: boolean;
  /** Max characters (for display warning) */
  maxChars?: number;
}

// Mode configuration
const MODE_CONFIG = {
  simple: { icon: "📝", color: "dim", label: "simple" },
  nlp: { icon: "✨", color: "magenta", label: "nlp" },
  orchestrated: { icon: "🔧", color: "cyan", label: "orch" },
};

export const InputPrompt: React.FC<InputPromptProps> = ({
  prompt = ">> ",
  placeholder = "Type a message... (Tab for suggestions, ↑↓ for history)",
  onSubmit,
  onExit,
  onInterrupt,
  disabled = false,
  processing = false,
  history = [],
  promptColor = "cyan",
  showBorder = true,
  mode,
  showCharCount = false,
  maxChars = 4000,
}) => {
  const [value, setValue] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursorPosition, setCursorPosition] = useState(0);

  useInput(
    (input, key) => {
      if (disabled) return;

      // Handle special keys
      if (key.return) {
        // Submit on Enter
        if (value.trim()) {
          // Check for exit command
          if (value.trim().toLowerCase() === "/exit" || value.trim().toLowerCase() === "/quit") {
            onExit?.();
            return;
          }
          onSubmit(value.trim());
          setValue("");
          setCursorPosition(0);
          setHistoryIndex(-1);
        }
        return;
      }

      if (key.ctrl && input === "c") {
        // Ctrl+C to exit
        onExit?.();
        return;
      }

      if (key.escape) {
        // Escape to interrupt processing
        if (processing && onInterrupt) {
          onInterrupt();
        }
        return;
      }

      if (key.ctrl && input === "u") {
        // Ctrl+U to clear line
        setValue("");
        setCursorPosition(0);
        return;
      }

      if (key.ctrl && input === "a") {
        // Ctrl+A to move to start
        setCursorPosition(0);
        return;
      }

      if (key.ctrl && input === "e") {
        // Ctrl+E to move to end
        setCursorPosition(value.length);
        return;
      }

      if (key.ctrl && input === "w") {
        // Ctrl+W to delete word backward
        const beforeCursor = value.slice(0, cursorPosition);
        const afterCursor = value.slice(cursorPosition);
        const newBefore = beforeCursor.replace(/\S+\s*$/, "");
        setValue(newBefore + afterCursor);
        setCursorPosition(newBefore.length);
        return;
      }

      if (key.backspace || key.delete) {
        // Backspace
        if (cursorPosition > 0) {
          setValue((v) => v.slice(0, cursorPosition - 1) + v.slice(cursorPosition));
          setCursorPosition((p) => p - 1);
        }
        return;
      }

      if (key.upArrow) {
        // Navigate history up
        if (history.length > 0 && historyIndex < history.length - 1) {
          const newIndex = historyIndex + 1;
          setHistoryIndex(newIndex);
          const historyValue = history[history.length - 1 - newIndex];
          setValue(historyValue);
          setCursorPosition(historyValue.length);
        }
        return;
      }

      if (key.downArrow) {
        // Navigate history down
        if (historyIndex > 0) {
          const newIndex = historyIndex - 1;
          setHistoryIndex(newIndex);
          const historyValue = history[history.length - 1 - newIndex];
          setValue(historyValue);
          setCursorPosition(historyValue.length);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setValue("");
          setCursorPosition(0);
        }
        return;
      }

      if (key.leftArrow) {
        // Move cursor left
        if (key.ctrl) {
          // Ctrl+Left: move to previous word
          const beforeCursor = value.slice(0, cursorPosition);
          const match = beforeCursor.match(/\S+\s*$/);
          setCursorPosition(match ? cursorPosition - match[0].length : 0);
        } else {
          setCursorPosition((p) => Math.max(0, p - 1));
        }
        return;
      }

      if (key.rightArrow) {
        // Move cursor right
        if (key.ctrl) {
          // Ctrl+Right: move to next word
          const afterCursor = value.slice(cursorPosition);
          const match = afterCursor.match(/^\s*\S+/);
          setCursorPosition(match ? cursorPosition + match[0].length : value.length);
        } else {
          setCursorPosition((p) => Math.min(value.length, p + 1));
        }
        return;
      }

      // Home/End keys - check via key codes if available
      if ((key as any).home) {
        setCursorPosition(0);
        return;
      }

      if ((key as any).end) {
        setCursorPosition(value.length);
        return;
      }

      // Regular character input
      if (input && !key.ctrl && !key.meta) {
        setValue((v) => v.slice(0, cursorPosition) + input + v.slice(cursorPosition));
        setCursorPosition((p) => p + input.length);
      }
    },
    { isActive: !disabled }  // Input active even while processing
  );

  // Display value with cursor
  const displayValue = value || "";
  const beforeCursor = displayValue.slice(0, cursorPosition);
  const atCursor = displayValue[cursorPosition] || " ";
  const afterCursor = displayValue.slice(cursorPosition + 1);

  // Character count warning
  const charWarning = value.length > maxChars * 0.9;

  const content = (
    <Box flexDirection="column">
      {/* Input row */}
      <Box>
        {/* Mode indicator */}
        {mode && (
          <Text color={MODE_CONFIG[mode].color}>
            {MODE_CONFIG[mode].icon}{" "}
          </Text>
        )}

        {/* Prompt */}
        <Text color={promptColor} bold>
          {prompt}
        </Text>

        {/* Input field */}
        {!disabled ? (
          <>
            <Text>{beforeCursor}</Text>
            <Text inverse>{atCursor}</Text>
            <Text>{afterCursor}</Text>
            {!value && (
              <Text color="dim">
                {processing
                  ? "Type to send follow-up, Esc to interrupt..."
                  : placeholder}
              </Text>
            )}
          </>
        ) : (
          <Text color="yellow">
            ◐ Processing...
          </Text>
        )}
      </Box>

      {/* Status row (char count, hints) */}
      {(showCharCount || history.length > 0) && !disabled && (
        <Box marginTop={1} justifyContent="space-between">
          <Box>
            {history.length > 0 && historyIndex === -1 && (
              <Text color="dim">↑↓ history ({history.length})</Text>
            )}
            {historyIndex >= 0 && (
              <Text color="dim">
                [{history.length - historyIndex}/{history.length}]
              </Text>
            )}
          </Box>
          {showCharCount && (
            <Text color={charWarning ? "yellow" : "dim"}>
              {value.length}/{maxChars}
            </Text>
          )}
        </Box>
      )}
    </Box>
  );

  if (showBorder) {
    return (
      <Box
        borderStyle="round"
        borderColor={disabled ? "dim" : processing ? "yellow" : promptColor}
        paddingX={1}
        paddingY={0}
      >
        {content}
      </Box>
    );
  }

  return content;
};

/**
 * Simple command input for slash commands.
 */
export interface CommandInputProps {
  onCommand: (cmd: string, args: string[]) => void;
  disabled?: boolean;
  commands?: string[];
}

export const CommandInput: React.FC<CommandInputProps> = ({
  onCommand,
  disabled = false,
  commands = ["help", "clear", "new", "simple", "exit"],
}) => {
  const handleSubmit = (value: string) => {
    if (!value.startsWith("/")) return;
    const parts = value.slice(1).split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);
    onCommand(cmd, args);
  };

  return (
    <InputPrompt
      prompt="/"
      promptColor="yellow"
      placeholder={`Commands: ${commands.join(", ")}`}
      onSubmit={handleSubmit}
      disabled={disabled}
      showBorder={false}
    />
  );
};

export default InputPrompt;
