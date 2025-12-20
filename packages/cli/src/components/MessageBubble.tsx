/**
 * MessageBubble component - Raw Document Style
 */

import React from "react";
import { Box, Text } from "ink";
import { MarkdownText } from "./MarkdownText.js";

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
}

export interface MessageBubbleProps {
  message: Message;
  streaming?: boolean;
}

const UserBubble: React.FC<{ message: Message }> = ({ message }) => {
  return (
    <Box marginY={1}>
      <Text color="blue" bold>{message.content}</Text>
    </Box>
  );
};

const AssistantBubble: React.FC<{
  message: Message;
  streaming?: boolean;
}> = ({ message, streaming }) => {
  return (
    <Box marginY={1} flexDirection="column">
      {streaming && !message.content ? (
        <Text color="dim">...</Text>
      ) : (
        <MarkdownText>{message.content}</MarkdownText>
      )}
    </Box>
  );
};

const SystemBubble: React.FC<{ message: Message }> = ({ message }) => {
  return (
    <Box marginY={1}>
      <Text color="dim" italic>{message.content}</Text>
    </Box>
  );
};

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  streaming = false,
}) => {
  switch (message.role) {
    case "user":
      return <UserBubble message={message} />;
    case "assistant":
      return <AssistantBubble message={message} streaming={streaming} />;
    case "system":
      return <SystemBubble message={message} />;
  }
};

export const StreamingBubble: React.FC<{ content: string }> = ({ content }) => {
  return (
    <Box marginY={1}>
      {content ? (
        <MarkdownText>{content}</MarkdownText>
      ) : (
        <Text color="dim">⠋</Text>
      )}
    </Box>
  );
};

export default MessageBubble;
