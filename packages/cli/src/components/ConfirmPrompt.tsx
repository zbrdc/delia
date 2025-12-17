/**
 * ConfirmPrompt component for confirming dangerous tool executions.
 *
 * Features:
 * - Yellow warning box for visibility
 * - Shows tool name and arguments
 * - Interactive prompt with keyboard shortcuts
 * - Support for "allow all" to skip future prompts
 */

import React, { useState, useCallback } from "react";
import { Box, Text, useInput } from "ink";

export interface ConfirmPromptProps {
  confirmId: string;
  tool: string;
  args: Record<string, unknown>;
  message: string;
  onConfirm: (confirmId: string, confirmed: boolean, allowAll: boolean) => void;
}

/**
 * Format tool arguments for display, showing more detail than ToolCallPanel.
 */
function formatArgsDetailed(args: Record<string, unknown>): React.ReactNode[] {
  return Object.entries(args).map(([key, value], index) => {
    const valueStr = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    // Truncate very long values but show more than in tool panel
    const displayValue = valueStr.length > 200 ? valueStr.slice(0, 200) + "..." : valueStr;

    return (
      <Box key={key} marginLeft={2}>
        <Text color="cyan">{key}</Text>
        <Text color="dim">: </Text>
        <Text>{displayValue}</Text>
      </Box>
    );
  });
}

/**
 * Interactive confirmation prompt for dangerous operations.
 */
export const ConfirmPrompt: React.FC<ConfirmPromptProps> = ({
  confirmId,
  tool,
  args,
  message,
  onConfirm,
}) => {
  const [responded, setResponded] = useState(false);

  const handleResponse = useCallback(
    (confirmed: boolean, allowAll: boolean = false) => {
      if (responded) return;
      setResponded(true);
      onConfirm(confirmId, confirmed, allowAll);
    },
    [confirmId, responded, onConfirm]
  );

  // Handle keyboard input
  useInput(
    (input, key) => {
      if (responded) return;

      // y = yes/confirm
      if (input.toLowerCase() === "y") {
        handleResponse(true, false);
      }
      // n = no/deny
      else if (input.toLowerCase() === "n" || key.escape) {
        handleResponse(false, false);
      }
      // a = allow all (yes + don't ask again)
      else if (input.toLowerCase() === "a") {
        handleResponse(true, true);
      }
    },
    { isActive: !responded }
  );

  if (responded) {
    return null;
  }

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="yellow"
      paddingX={2}
      paddingY={1}
      marginY={1}
    >
      {/* Warning header */}
      <Box marginBottom={1}>
        <Text color="yellow" bold>
          [!] Confirmation Required
        </Text>
      </Box>

      {/* Tool info */}
      <Box flexDirection="column" marginBottom={1}>
        <Box>
          <Text>Tool: </Text>
          <Text bold color="yellow">
            {tool}
          </Text>
        </Box>

        {/* Arguments */}
        {Object.keys(args).length > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <Text color="dim">Arguments:</Text>
            {formatArgsDetailed(args)}
          </Box>
        )}
      </Box>

      {/* Prompt */}
      <Box marginTop={1} borderStyle="single" borderColor="dim" paddingX={1}>
        <Text>
          <Text color="yellow">[y]</Text>
          <Text color="dim"> yes  </Text>
          <Text color="red">[n]</Text>
          <Text color="dim"> no  </Text>
          <Text color="green">[a]</Text>
          <Text color="dim"> allow all (don't ask again)</Text>
        </Text>
      </Box>
    </Box>
  );
};

export default ConfirmPrompt;
