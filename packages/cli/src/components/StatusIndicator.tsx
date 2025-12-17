/**
 * StatusIndicator component for displaying transient status messages.
 *
 * Shows brief status updates that disappear when the phase completes.
 * Designed for a clean, coding-focused experience.
 */

import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";

export interface StatusDetails {
  tier?: string;
  model?: string;
  backend?: string;
  voting_k?: number;
  votes_cast?: number;
  red_flagged?: number;
  quality_score?: number;
  task_type?: string;
}

export interface StatusInfo {
  phase: "routing" | "voting" | "quality" | "model" | "thinking" | "processing";
  message: string;
  details?: StatusDetails;
}

export interface StatusIndicatorProps {
  status: StatusInfo | null;
}

/**
 * Simple transient status - shows message with spinner, disappears when done.
 */
export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  if (!status) return null;

  return (
    <Box marginY={1}>
      <Text color="dim">
        <Spinner type="dots" />
      </Text>
      <Text color="dim"> {status.message}</Text>
      {status.details?.model && (
        <Text color="cyan"> {status.details.model}</Text>
      )}
    </Box>
  );
};

/**
 * Tug-of-war voting progress bar for k-voting.
 * Visual representation: consensus pulls left (green), disagreement pulls right (red).
 * The bar shows the "tension" between agreeing and disagreeing votes.
 */
export interface VotingProgressProps {
  /** Total votes needed (k) */
  totalVotes: number;
  /** Votes cast so far */
  votesCast: number;
  /** Number of agreeing votes */
  consensus?: number;
  /** Number of flagged/disagreeing votes */
  flagged?: number;
  /** Width of the bar */
  width?: number;
}

export const VotingProgress: React.FC<VotingProgressProps> = ({
  totalVotes,
  votesCast,
  consensus = 0,
  flagged = 0,
  width = 24,
}) => {
  // Tug-of-war visualization:
  // Center is neutral, green pulls left, red pulls right
  const halfWidth = Math.floor(width / 2);
  
  // Calculate the "pull" - positive means consensus winning, negative means flagged winning
  const total = consensus + flagged;
  const pull = total > 0 ? (consensus - flagged) / total : 0;
  
  // Position of the "marker" from center (-halfWidth to +halfWidth)
  const markerOffset = Math.round(pull * halfWidth);
  
  // Build the bar
  const leftSide: string[] = [];
  const rightSide: string[] = [];
  
  // Left side (consensus territory)
  for (let i = 0; i < halfWidth; i++) {
    const distFromCenter = halfWidth - i;
    if (markerOffset >= distFromCenter) {
      leftSide.push("█"); // Consensus territory
    } else {
      leftSide.push("░");
    }
  }
  
  // Right side (flagged territory)
  for (let i = 0; i < halfWidth; i++) {
    const distFromCenter = i + 1;
    if (-markerOffset >= distFromCenter) {
      rightSide.push("█"); // Flagged territory
    } else {
      rightSide.push("░");
    }
  }

  // Determine center marker based on current state
  const centerMarker = votesCast === 0 ? "│" : pull > 0 ? "◀" : pull < 0 ? "▶" : "│";
  const centerColor = pull > 0 ? "green" : pull < 0 ? "red" : "dim";

  return (
    <Box marginY={1}>
      <Text color="green">✓{consensus} </Text>
      <Text color="green">{leftSide.join("")}</Text>
      <Text color={centerColor} bold>{centerMarker}</Text>
      <Text color="red">{rightSide.join("")}</Text>
      <Text color="red"> ✗{flagged}</Text>
      <Text color="dim"> ({votesCast}/{totalVotes})</Text>
    </Box>
  );
};

/**
 * Compact status badge for headers.
 */
export const StatusBadge: React.FC<{
  phase: StatusInfo["phase"];
  animated?: boolean;
}> = ({ phase, animated }) => {
  const icons: Record<string, string> = {
    routing: "→",
    voting: "🗳️",
    quality: "★",
    model: "🍈",
    thinking: "🍈",
    processing: "🍈",
  };

  return (
    <Text color="dim">
      {animated && <Spinner type="dots" />}
      {!animated && icons[phase]}
    </Text>
  );
};

export default StatusIndicator;
