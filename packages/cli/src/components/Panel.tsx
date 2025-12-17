/**
 * Panel component for consistent bordered containers.
 *
 * Provides styled containers with various border styles and colors.
 */

import React from "react";
import { Box, Text, BoxProps } from "ink";

export interface PanelProps {
  /** Panel title (shown in top border) */
  title?: string;
  /** Title color */
  titleColor?: string;
  /** Border style */
  borderStyle?: BoxProps["borderStyle"];
  /** Border color */
  borderColor?: string;
  /** Padding inside panel */
  padding?: number;
  /** Panel width */
  width?: number | string;
  /** Children components */
  children: React.ReactNode;
  /** Collapsed state (show only title) */
  collapsed?: boolean;
  /** Icon to show before title */
  icon?: string;
  /** Right-side content in header */
  headerRight?: React.ReactNode;
  /** Status indicator color */
  status?: "success" | "error" | "warning" | "info" | "none";
}

const STATUS_COLORS: Record<string, string> = {
  success: "green",
  error: "red",
  warning: "yellow",
  info: "blue",
  none: "dim",
};

const STATUS_ICONS: Record<string, string> = {
  success: "✓",
  error: "✗",
  warning: "⚠",
  info: "ℹ",
  none: "",
};

export const Panel: React.FC<PanelProps> = ({
  title,
  titleColor = "cyan",
  borderStyle = "round",
  borderColor = "dim",
  padding = 1,
  width,
  children,
  collapsed = false,
  icon,
  headerRight,
  status = "none",
}) => {
  const statusColor = STATUS_COLORS[status];
  const statusIcon = STATUS_ICONS[status];

  return (
    <Box
      flexDirection="column"
      borderStyle={borderStyle}
      borderColor={status !== "none" ? statusColor : borderColor}
      paddingX={padding}
      paddingY={collapsed ? 0 : padding}
      width={width}
    >
      {/* Header row */}
      {(title || headerRight) && (
        <Box justifyContent="space-between" marginBottom={collapsed ? 0 : 1}>
          <Box>
            {icon && <Text>{icon} </Text>}
            {status !== "none" && (
              <Text color={statusColor}>{statusIcon} </Text>
            )}
            {title && (
              <Text color={titleColor} bold>
                {title}
              </Text>
            )}
          </Box>
          {headerRight}
        </Box>
      )}

      {/* Content */}
      {!collapsed && children}
    </Box>
  );
};

/**
 * Divider line for separating sections.
 */
export interface DividerProps {
  /** Character to use for line */
  char?: string;
  /** Line color */
  color?: string;
  /** Text to show in middle of divider */
  text?: string;
  /** Width of divider (defaults to 60) */
  width?: number;
}

export const Divider: React.FC<DividerProps> = ({
  char = "─",
  color = "dim",
  text,
  width = 60,
}) => {
  if (text) {
    const textLen = text.length + 2; // padding
    const sideLen = Math.max(2, Math.floor((width - textLen) / 2));
    return (
      <Box marginY={1}>
        <Text color={color}>{char.repeat(sideLen)} </Text>
        <Text color={color}>{text}</Text>
        <Text color={color}> {char.repeat(sideLen)}</Text>
      </Box>
    );
  }

  return (
    <Box marginY={1}>
      <Text color={color}>{char.repeat(width)}</Text>
    </Box>
  );
};

/**
 * Card component - a simpler panel variant.
 */
export interface CardProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "error" | "warning" | "info";
  compact?: boolean;
}

const CARD_COLORS: Record<string, string> = {
  default: "dim",
  success: "green",
  error: "red",
  warning: "yellow",
  info: "blue",
};

export const Card: React.FC<CardProps> = ({
  children,
  variant = "default",
  compact = false,
}) => {
  return (
    <Box
      borderStyle="single"
      borderColor={CARD_COLORS[variant]}
      paddingX={compact ? 1 : 2}
      paddingY={compact ? 0 : 1}
    >
      {children}
    </Box>
  );
};

/**
 * Badge - inline status indicator.
 */
export interface BadgeProps {
  children: React.ReactNode;
  color?: string;
  backgroundColor?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  color = "white",
  backgroundColor,
}) => {
  return (
    <Text color={color} backgroundColor={backgroundColor} bold>
      [{children}]
    </Text>
  );
};

/**
 * Chip - small labeled value.
 */
export interface ChipProps {
  label: string;
  value?: string | number;
  color?: string;
}

export const Chip: React.FC<ChipProps> = ({ label, value, color = "cyan" }) => {
  return (
    <Text>
      <Text color="dim">{label}:</Text>
      <Text color={color}>{value}</Text>
    </Text>
  );
};

export default Panel;
