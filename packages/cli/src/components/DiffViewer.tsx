/**
 * DiffViewer component for displaying file diffs.
 *
 * Shows additions, deletions, and context lines with color coding.
 */

import React from "react";
import { Box, Text } from "ink";

export interface DiffLine {
  type: "add" | "remove" | "context" | "header";
  content: string;
  lineNumber?: number;
  oldLineNumber?: number;
  newLineNumber?: number;
}

export interface DiffViewerProps {
  /** File path being diffed */
  filePath?: string;
  /** Raw unified diff text */
  diff?: string;
  /** Pre-parsed diff lines */
  lines?: DiffLine[];
  /** Maximum lines to display */
  maxLines?: number;
  /** Show line numbers */
  showLineNumbers?: boolean;
}

/**
 * Parse unified diff format into structured lines.
 */
function parseDiff(diff: string): DiffLine[] {
  const lines: DiffLine[] = [];
  const rawLines = diff.split("\n");

  let oldLine = 0;
  let newLine = 0;

  for (const line of rawLines) {
    // File headers
    if (line.startsWith("---") || line.startsWith("+++")) {
      lines.push({ type: "header", content: line });
      continue;
    }

    // Hunk header @@ -1,5 +1,6 @@
    if (line.startsWith("@@")) {
      lines.push({ type: "header", content: line });
      // Parse line numbers
      const match = line.match(/@@ -(\d+),?\d* \+(\d+),?\d* @@/);
      if (match) {
        oldLine = parseInt(match[1], 10);
        newLine = parseInt(match[2], 10);
      }
      continue;
    }

    // Addition
    if (line.startsWith("+")) {
      lines.push({
        type: "add",
        content: line.slice(1),
        newLineNumber: newLine++,
      });
      continue;
    }

    // Removal
    if (line.startsWith("-")) {
      lines.push({
        type: "remove",
        content: line.slice(1),
        oldLineNumber: oldLine++,
      });
      continue;
    }

    // Context line
    if (line.startsWith(" ") || line === "") {
      lines.push({
        type: "context",
        content: line.slice(1) || "",
        oldLineNumber: oldLine++,
        newLineNumber: newLine++,
      });
      continue;
    }

    // Other (comments, etc.)
    lines.push({ type: "context", content: line });
  }

  return lines;
}

const DiffLineComponent: React.FC<{
  line: DiffLine;
  showLineNumbers: boolean;
}> = ({ line, showLineNumbers }) => {
  const colors = {
    add: "green",
    remove: "red",
    context: undefined,
    header: "cyan",
  } as const;

  const prefixes = {
    add: "+",
    remove: "-",
    context: " ",
    header: "",
  };

  const bgColors = {
    add: "greenBright",
    remove: "redBright",
    context: undefined,
    header: undefined,
  };

  const lineNum = showLineNumbers
    ? line.type === "add"
      ? `    ${String(line.newLineNumber || "").padStart(4)} `
      : line.type === "remove"
      ? `${String(line.oldLineNumber || "").padStart(4)}     `
      : `${String(line.oldLineNumber || "").padStart(4)} ${String(line.newLineNumber || "").padStart(4)} `
    : "";

  return (
    <Box>
      {showLineNumbers && (
        <Text color="dim">{lineNum}</Text>
      )}
      <Text color={colors[line.type]} backgroundColor={bgColors[line.type]}>
        {prefixes[line.type]}
        {line.content}
      </Text>
    </Box>
  );
};

export const DiffViewer: React.FC<DiffViewerProps> = ({
  filePath,
  diff,
  lines: preLines,
  showLineNumbers = true,
}) => {
  // Parse diff if not pre-parsed
  const lines = preLines || (diff ? parseDiff(diff) : []);

  // Count additions and deletions
  const additions = lines.filter((l) => l.type === "add").length;
  const deletions = lines.filter((l) => l.type === "remove").length;

  return (
    <Box flexDirection="column" marginY={1}>
      {/* Header */}
      <Box>
        {filePath && (
          <Text bold color="cyan">
            {filePath}
          </Text>
        )}
        <Text color="dim">
          {" "}
          (<Text color="green">+{additions}</Text>
          {" / "}
          <Text color="red">-{deletions}</Text>)
        </Text>
      </Box>

      {/* Diff content */}
      <Box
        flexDirection="column"
        paddingX={2}
      >
        {lines.map((line, i) => (
          <DiffLineComponent
            key={i}
            line={line}
            showLineNumbers={showLineNumbers}
          />
        ))}
      </Box>
    </Box>
  );
};

/**
 * Simple inline diff for showing small changes.
 */
export const InlineDiff: React.FC<{
  before: string;
  after: string;
}> = ({ before, after }) => {
  return (
    <Box>
      <Text color="red" strikethrough>
        {before}
      </Text>
      <Text> → </Text>
      <Text color="green">{after}</Text>
    </Box>
  );
};
