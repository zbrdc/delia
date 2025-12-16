/**
 * StatusIndicator component for displaying advanced logic stats.
 *
 * Shows a neat status bar with k-voting, routing, and quality info
 * that appears during processing and disappears when done.
 */

import React from "react";
import { Box, Text } from "ink";

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
  phase: "routing" | "voting" | "quality" | "model";
  message: string;
  details?: StatusDetails;
}

export interface StatusIndicatorProps {
  status: StatusInfo | null;
}

// Color scheme for different phases
const PHASE_COLORS: Record<string, string> = {
  routing: "cyan",
  voting: "magenta",
  quality: "yellow",
  model: "blue",
};

// Icons for phases (ASCII-safe)
const PHASE_ICONS: Record<string, string> = {
  routing: ">",
  voting: "#",
  quality: "*",
  model: "@",
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  if (!status) return null;

  const color = PHASE_COLORS[status.phase] || "white";
  const icon = PHASE_ICONS[status.phase] || ">";

  // Build detail chips
  const chips: string[] = [];
  const details = status.details || {};

  if (details.tier) {
    chips.push(`tier:${details.tier}`);
  }
  if (details.voting_k !== undefined) {
    const votesInfo = details.votes_cast !== undefined
      ? `${details.votes_cast}/${details.voting_k}`
      : `k=${details.voting_k}`;
    chips.push(`votes:${votesInfo}`);
  }
  if (details.red_flagged && details.red_flagged > 0) {
    chips.push(`flagged:${details.red_flagged}`);
  }
  if (details.quality_score !== undefined) {
    const score = (details.quality_score * 100).toFixed(0);
    chips.push(`quality:${score}%`);
  }
  if (details.backend) {
    chips.push(`backend:${details.backend}`);
  }

  return (
    <Box marginY={0} paddingX={1}>
      <Text color="dim">[</Text>
      <Text color={color} bold>{icon}</Text>
      <Text color="dim">]</Text>
      <Text color={color}> {status.message}</Text>
      {chips.length > 0 && (
        <>
          <Text color="dim"> | </Text>
          {chips.map((chip, i) => (
            <React.Fragment key={chip}>
              {i > 0 && <Text color="dim"> </Text>}
              <Text color="dim">{chip}</Text>
            </React.Fragment>
          ))}
        </>
      )}
    </Box>
  );
};

export default StatusIndicator;
