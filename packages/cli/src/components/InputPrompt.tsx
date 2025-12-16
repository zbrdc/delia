/**
 * InputPrompt component for interactive text input.
 *
 * Provides a readline-style input with history support.
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
  /** Whether input is disabled */
  disabled?: boolean;
  /** Input history for up/down navigation */
  history?: string[];
  /** Prompt color */
  promptColor?: string;
}

export const InputPrompt: React.FC<InputPromptProps> = ({
  prompt = "> ",
  placeholder = "Type a message...",
  onSubmit,
  onExit,
  disabled = false,
  history = [],
  promptColor = "cyan",
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

      if (key.ctrl && input === "u") {
        // Ctrl+U to clear line
        setValue("");
        setCursorPosition(0);
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
        setCursorPosition((p) => Math.max(0, p - 1));
        return;
      }

      if (key.rightArrow) {
        // Move cursor right
        setCursorPosition((p) => Math.min(value.length, p + 1));
        return;
      }

      // Regular character input
      if (input && !key.ctrl && !key.meta) {
        setValue((v) => v.slice(0, cursorPosition) + input + v.slice(cursorPosition));
        setCursorPosition((p) => p + input.length);
      }
    },
    { isActive: !disabled }
  );

  // Display value with cursor
  const displayValue = value || "";
  const beforeCursor = displayValue.slice(0, cursorPosition);
  const atCursor = displayValue[cursorPosition] || " ";
  const afterCursor = displayValue.slice(cursorPosition + 1);

  return (
    <Box>
      <Text color={promptColor} bold>
        {prompt}
      </Text>
      {!disabled ? (
        <>
          <Text>{beforeCursor}</Text>
          <Text inverse>{atCursor}</Text>
          <Text>{afterCursor}</Text>
          {!value && (
            <Text color="dim">{placeholder}</Text>
          )}
        </>
      ) : (
        <Text color="dim">Processing...</Text>
      )}
    </Box>
  );
};

export default InputPrompt;
