/**
 * MarkdownText component for terminal markdown rendering.
 *
 * Renders markdown content with appropriate terminal formatting:
 * - Headers with bold/colors
 * - Code blocks with syntax highlighting
 * - Inline code with background
 * - Lists with bullets
 * - Bold and italic text
 */

import React from "react";
import { Box, Text } from "ink";
import { highlight, supportsLanguage } from "cli-highlight";

export interface MarkdownTextProps {
  children: string;
}

interface ParsedBlock {
  type: "paragraph" | "heading" | "code" | "list" | "blockquote" | "hr";
  content: string;
  level?: number; // For headings (1-6)
  language?: string; // For code blocks
  items?: string[]; // For lists
  ordered?: boolean; // For lists
}

/**
 * Parse markdown into blocks.
 */
function parseBlocks(markdown: string): ParsedBlock[] {
  const blocks: ParsedBlock[] = [];
  const lines = markdown.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Code block (```)
    if (line.startsWith("```")) {
      const language = line.slice(3).trim() || undefined;
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      blocks.push({
        type: "code",
        content: codeLines.join("\n"),
        language,
      });
      i++;
      continue;
    }

    // Heading (#)
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        content: headingMatch[2],
        level: headingMatch[1].length,
      });
      i++;
      continue;
    }

    // Horizontal rule (---, ***, ___)
    if (/^[-*_]{3,}$/.test(line.trim())) {
      blocks.push({ type: "hr", content: "" });
      i++;
      continue;
    }

    // Unordered list (-, *, +)
    if (/^[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*+]\s+/, ""));
        i++;
      }
      blocks.push({
        type: "list",
        content: "",
        items,
        ordered: false,
      });
      continue;
    }

    // Ordered list (1., 2., etc.)
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ""));
        i++;
      }
      blocks.push({
        type: "list",
        content: "",
        items,
        ordered: true,
      });
      continue;
    }

    // Blockquote (>)
    if (line.startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        quoteLines.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      blocks.push({
        type: "blockquote",
        content: quoteLines.join("\n"),
      });
      continue;
    }

    // Empty line - skip
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph - collect until empty line or special block
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].startsWith("```") &&
      !lines[i].startsWith("#") &&
      !/^[-*+]\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !lines[i].startsWith(">") &&
      !/^[-*_]{3,}$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({
        type: "paragraph",
        content: paraLines.join(" "),
      });
    }
  }

  return blocks;
}

/**
 * Render inline markdown (bold, italic, code, links).
 */
const InlineMarkdown: React.FC<{ text: string }> = ({ text }) => {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Inline code `code`
    let match = remaining.match(/^`([^`]+)`/);
    if (match) {
      parts.push(
        <Text key={key++} backgroundColor="gray" color="white">
          {` ${match[1]} `}
        </Text>
      );
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Bold **text** or __text__
    match = remaining.match(/^(\*\*|__)([^*_]+)\1/);
    if (match) {
      parts.push(
        <Text key={key++} bold>
          {match[2]}
        </Text>
      );
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Italic *text* or _text_
    match = remaining.match(/^(\*|_)([^*_]+)\1/);
    if (match) {
      parts.push(
        <Text key={key++} italic>
          {match[2]}
        </Text>
      );
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Link [text](url)
    match = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/);
    if (match) {
      parts.push(
        <Text key={key++} color="blue" underline>
          {match[1]}
        </Text>
      );
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Plain text until next special char
    match = remaining.match(/^[^`*_\[]+/);
    if (match) {
      parts.push(<Text key={key++}>{match[0]}</Text>);
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Single special char (not part of formatting)
    parts.push(<Text key={key++}>{remaining[0]}</Text>);
    remaining = remaining.slice(1);
  }

  return <Text>{parts}</Text>;
};

/**
 * Render a code block with syntax highlighting.
 */
const CodeBlockRender: React.FC<{
  code: string;
  language?: string;
}> = ({ code, language }) => {
  let highlighted = code;

  if (language && supportsLanguage(language)) {
    try {
      highlighted = highlight(code, { language, ignoreIllegals: true });
    } catch {
      // Use plain code on error
    }
  }

  return (
    <Box flexDirection="column" marginY={1}>
      {language && (
        <Text color="dim" italic>
          {language}
        </Text>
      )}
      <Box paddingX={2}>
        <Text>{highlighted}</Text>
      </Box>
    </Box>
  );
};

export const MarkdownText: React.FC<MarkdownTextProps> = ({
  children,
}) => {
  const blocks = parseBlocks(children);

  return (
    <Box flexDirection="column">
      {blocks.map((block, i) => {
        switch (block.type) {
          case "heading": {
            const colors = ["magenta", "cyan", "blue", "green", "yellow", "white"] as const;
            const color = colors[Math.min((block.level || 1) - 1, colors.length - 1)];
            return (
              <Box key={i} marginY={1}>
                <Text bold color={color}>
                  {"#".repeat(block.level || 1)} {block.content}
                </Text>
              </Box>
            );
          }

          case "code":
            return (
              <CodeBlockRender
                key={i}
                code={block.content}
                language={block.language}
              />
            );

          case "list":
            return (
              <Box key={i} flexDirection="column" marginY={1}>
                {block.items?.map((item, j) => (
                  <Box key={j}>
                    <Text color="dim">
                      {block.ordered ? `${j + 1}. ` : "  - "}
                    </Text>
                    <InlineMarkdown text={item} />
                  </Box>
                ))}
              </Box>
            );

          case "blockquote":
            return (
              <Box key={i} marginY={1} paddingLeft={2}>
                <Text color="dim">│ </Text>
                <Text italic color="gray">
                  {block.content}
                </Text>
              </Box>
            );

          case "hr":
            return (
              <Box key={i} marginY={1}>
                <Text color="dim">{"─".repeat(40)}</Text>
              </Box>
            );

          case "paragraph":
          default:
            return (
              <Box key={i} marginY={1}>
                <InlineMarkdown text={block.content} />
              </Box>
            );
        }
      })}
    </Box>
  );
};
