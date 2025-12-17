/**
 * ProgressBar and status indicator components.
 *
 * Provides visual feedback for ongoing operations.
 */

import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";

/**
 * Progress bar with percentage and optional label.
 */
export interface ProgressBarProps {
  /** Progress value (0-100) */
  value: number;
  /** Total width of bar */
  width?: number;
  /** Show percentage text */
  showPercent?: boolean;
  /** Label to show before bar */
  label?: string;
  /** Fill character */
  fillChar?: string;
  /** Empty character */
  emptyChar?: string;
  /** Bar color (changes based on progress) */
  color?: string;
  /** Use gradient colors based on progress */
  gradient?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  width = 30,
  showPercent = true,
  label,
  fillChar = "█",
  emptyChar = "░",
  color,
  gradient = true,
}) => {
  const percent = Math.max(0, Math.min(100, value));
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;

  // Determine color based on progress
  let barColor = color;
  if (gradient && !color) {
    if (percent < 33) barColor = "red";
    else if (percent < 66) barColor = "yellow";
    else barColor = "green";
  }

  return (
    <Box>
      {label && <Text color="dim">{label} </Text>}
      <Text color={barColor}>
        {fillChar.repeat(filled)}
      </Text>
      <Text color="dim">{emptyChar.repeat(empty)}</Text>
      {showPercent && (
        <Text color="dim"> {percent.toFixed(0)}%</Text>
      )}
    </Box>
  );
};

/**
 * Animated spinner with label.
 */
export interface SpinnerLabelProps {
  /** Label text */
  label: string;
  /** Spinner type */
  type?: "dots" | "dots2" | "dots3" | "line" | "arc" | "circle" | "bouncingBar";
  /** Color */
  color?: string;
}

export const SpinnerLabel: React.FC<SpinnerLabelProps> = ({
  label,
  type = "dots",
  color = "cyan",
}) => {
  return (
    <Box>
      <Text color={color}>
        <Spinner type={type} />
      </Text>
      <Text> {label}</Text>
    </Box>
  );
};

/**
 * Spinning melon animation frames.
 */
const MELON_FRAMES = ["🍈", "🍈", "🍈", "🍈"];
const SPIN_FRAMES = ["◐", "◓", "◑", "◒"];

/**
 * Thinking/processing indicator with spinning melon.
 */
export interface ThinkingIndicatorProps {
  status: string;
  variant?: "melon" | "spinner" | "dots" | "pulse";
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({
  status,
  variant = "melon",
}) => {
  const [frame, setFrame] = useState(0);
  const [dotCount, setDotCount] = useState(1);

  useEffect(() => {
    if (variant === "dots") {
      const interval = setInterval(() => {
        setDotCount((c) => (c % 3) + 1);
      }, 400);
      return () => clearInterval(interval);
    }
    
    if (variant === "melon" || variant === "spinner" || variant === "pulse") {
      const interval = setInterval(() => {
        setFrame((f) => (f + 1) % SPIN_FRAMES.length);
      }, 150);
      return () => clearInterval(interval);
    }
  }, [variant]);

  if (variant === "melon") {
    return (
      <Box marginY={1}>
        <Text color="green">🍈</Text>
        <Text color="yellow">{SPIN_FRAMES[frame]} </Text>
        <Text color="yellow">{status}</Text>
      </Box>
    );
  }

  if (variant === "spinner") {
    return (
      <Box marginY={1}>
        <Text color="yellow">
          <Spinner type="dots" /> {status}
        </Text>
      </Box>
    );
  }

  if (variant === "dots") {
    return (
      <Box marginY={1}>
        <Text color="yellow">
          {status}{".".repeat(dotCount)}{" ".repeat(3 - dotCount)}
        </Text>
      </Box>
    );
  }

  // Pulse variant
  return (
    <Box marginY={1}>
      <Text color="yellow">
        <Spinner type="arc" /> {status}
      </Text>
    </Box>
  );
};

/**
 * Step progress indicator for multi-step operations.
 */
export interface Step {
  label: string;
  status: "pending" | "running" | "done" | "error";
}

export interface StepProgressProps {
  steps: Step[];
  currentStep?: number;
  showLabels?: boolean;
}

const STEP_ICONS = {
  pending: "○",
  running: "◐",
  done: "●",
  error: "✗",
};

const STEP_COLORS = {
  pending: "dim",
  running: "yellow",
  done: "green",
  error: "red",
};

export const StepProgress: React.FC<StepProgressProps> = ({
  steps,
  currentStep,
  showLabels = true,
}) => {
  return (
    <Box flexDirection="column">
      {/* Progress dots */}
      <Box>
        {steps.map((step, i) => (
          <React.Fragment key={i}>
            <Text color={STEP_COLORS[step.status]}>
              {step.status === "running" ? (
                <Spinner type="dots" />
              ) : (
                STEP_ICONS[step.status]
              )}
            </Text>
            {i < steps.length - 1 && (
              <Text color="dim">───</Text>
            )}
          </React.Fragment>
        ))}
      </Box>

      {/* Labels below dots */}
      {showLabels && (
        <Box marginTop={1}>
          {steps.map((step, i) => (
            <Box key={i} width={8}>
              <Text
                color={STEP_COLORS[step.status]}
                wrap="truncate"
              >
                {step.label}
              </Text>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
};

/**
 * Iteration counter for agentic loops.
 */
export interface IterationCounterProps {
  current: number;
  max?: number;
  label?: string;
}

export const IterationCounter: React.FC<IterationCounterProps> = ({
  current,
  max,
  label = "Iteration",
}) => {
  return (
    <Box>
      <Text color="cyan" bold>
        {label}
      </Text>
      <Text color="dim"> </Text>
      <Text color="cyan">{current}</Text>
      {max && (
        <>
          <Text color="dim">/</Text>
          <Text color="dim">{max}</Text>
        </>
      )}
    </Box>
  );
};

/**
 * Animated typing indicator.
 */
export const TypingIndicator: React.FC = () => {
  const [frame, setFrame] = useState(0);
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame((f) => (f + 1) % frames.length);
    }, 80);
    return () => clearInterval(interval);
  }, []);

  return (
    <Text color="dim">{frames[frame]} typing...</Text>
  );
};

/**
 * Token counter with rate.
 */
export interface TokenCounterProps {
  tokens: number;
  elapsed_ms?: number;
  showRate?: boolean;
}

export const TokenCounter: React.FC<TokenCounterProps> = ({
  tokens,
  elapsed_ms,
  showRate = true,
}) => {
  const rate = elapsed_ms && elapsed_ms > 0 
    ? (tokens / (elapsed_ms / 1000)).toFixed(1) 
    : null;

  return (
    <Box>
      <Text color="dim">{tokens} tok</Text>
      {showRate && rate && (
        <Text color="dim"> ({rate} tok/s)</Text>
      )}
    </Box>
  );
};

export default ProgressBar;
