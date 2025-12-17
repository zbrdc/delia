/**
 * CodeBlock component with syntax highlighting.
 *
 * Uses cli-highlight for terminal-friendly syntax highlighting.
 */

import React from "react";
import { Box, Text } from "ink";
import { highlight, supportsLanguage } from "cli-highlight";

export interface CodeBlockProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
  maxLines?: number;
  title?: string;
}

/**
 * Detect language from code content or filename.
 */
function detectLanguage(code: string, hint?: string): string {
  if (hint && supportsLanguage(hint)) {
    return hint;
  }

  // Simple heuristics for common languages
  if (code.includes("def ") && code.includes(":")) return "python";
  if (code.includes("function ") || code.includes("const ") || code.includes("=>")) return "javascript";
  if (code.includes("interface ") || code.includes(": string") || code.includes(": number")) return "typescript";
  if (code.includes("func ") && code.includes("package ")) return "go";
  if (code.includes("fn ") && code.includes("let ") && code.includes("mut ")) return "rust";
  if (code.includes("<?php")) return "php";
  if (code.includes("<html") || code.includes("<!DOCTYPE")) return "html";
  if (code.includes("SELECT ") || code.includes("INSERT ")) return "sql";
  if (code.startsWith("{") && code.includes(":")) return "json";
  if (code.includes("---") && code.includes(":")) return "yaml";

  return "plaintext";
}

/**
 * Apply syntax highlighting to code.
 */
function highlightCode(code: string, language: string): string {
  try {
    if (supportsLanguage(language)) {
      return highlight(code, { language, ignoreIllegals: true });
    }
    return code;
  } catch {
    return code;
  }
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language,
  showLineNumbers = false,
  maxLines,
  title,
}) => {
  const detectedLang = detectLanguage(code, language);
  const highlighted = highlightCode(code, detectedLang);

  // Split into lines - no truncation, show full content
  const lines = highlighted.split("\n");

  // Add line numbers if requested
  const lineNumberWidth = String(lines.length).length;
  const formattedLines = showLineNumbers
    ? lines.map((line, i) => {
        const num = String(i + 1).padStart(lineNumberWidth, " ");
        return `${num} │ ${line}`;
      })
    : lines;

  const content = formattedLines.join("\n");

  return (
    <Box flexDirection="column" marginY={1}>
      {title && (
        <Box>
          <Text color="cyan" bold>
            {title}
          </Text>
          <Text color="dim"> ({detectedLang})</Text>
        </Box>
      )}
      <Box
        paddingX={2}
        flexDirection="column"
      >
        <Text>{content}</Text>
      </Box>
    </Box>
  );
};

/**
 * Inline code span (no box, just highlighted).
 */
export const InlineCode: React.FC<{ children: string }> = ({ children }) => {
  return (
    <Text backgroundColor="gray" color="white">
      {` ${children} `}
    </Text>
  );
};
