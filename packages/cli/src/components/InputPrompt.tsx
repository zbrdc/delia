/**
 * InputPrompt component for interactive text input.
 *
 * Provides a clean readline-style input.
 */

import React, { useState, useCallback } from "react";
import { Box, Text, useInput } from "ink";

export interface InputPromptProps {
  prompt?: string;
  placeholder?: string;
  onSubmit: (value: string) => void;
  onExit?: () => void;
  onInterrupt?: () => void;
  disabled?: boolean;
  processing?: boolean;
  history?: string[];
  promptColor?: string;
  showBorder?: boolean;
  mode?: string;
  showCharCount?: boolean;
}

export const InputPrompt: React.FC<InputPromptProps> = ({
  prompt = "> ",
  onSubmit,
  onExit,
  onInterrupt,
  disabled = false,
  processing = false,
  history = [],
  promptColor = "blue",
}) => {
  const [value, setValue] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursorPosition, setCursorPosition] = useState(0);

  useInput(
    (input, key) => {
      if (disabled) return;

      if (key.return) {
        if (value.trim()) {
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
        onExit?.();
        return;
      }

      if (key.escape) {
        if (processing && onInterrupt) {
          onInterrupt();
        }
        return;
      }

      if (key.backspace || key.delete) {
        if (cursorPosition > 0) {
          setValue((v) => v.slice(0, cursorPosition - 1) + v.slice(cursorPosition));
          setCursorPosition((p) => p - 1);
        }
        return;
      }

      if (key.upArrow) {
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
        setCursorPosition((p) => Math.max(0, p - 1));
        return;
      }

      if (key.rightArrow) {
        setCursorPosition((p) => Math.min(value.length, p + 1));
        return;
      }

      if (input && !key.ctrl && !key.meta) {
        setValue((v) => v.slice(0, cursorPosition) + input + v.slice(cursorPosition));
        setCursorPosition((p) => p + input.length);
      }
    },
    { isActive: !disabled }
  );

  const displayValue = value || "";
  const beforeCursor = displayValue.slice(0, cursorPosition);
  const atCursor = displayValue[cursorPosition] || " ";
  const afterCursor = displayValue.slice(cursorPosition + 1);

  return (
    <Box flexDirection="row">
      <Text color={promptColor} bold>{prompt}</Text>
      
      {!disabled ? (
        <Text>
          <Text>{beforeCursor}</Text>
          <Text inverse>{atCursor}</Text>
          <Text>{afterCursor}</Text>
        </Text>
      ) : (
        <Text color="dim">...</Text>
      )}
    </Box>
  );
};

export default InputPrompt;