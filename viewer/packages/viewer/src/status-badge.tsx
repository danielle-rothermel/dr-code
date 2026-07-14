import type { JSX, ReactNode } from "react";

export type BadgeStatus = "positive" | "negative" | "warning" | "neutral";

export interface StatusBadgeProps {
  status?: BadgeStatus;
  children: ReactNode;
  className?: string;
}

/**
 * Small inline badge that colors its label by a generic status. Boring
 * on purpose: the status enum carries no domain meaning, so callers map
 * their own states (pass/fail, ok/error, …) onto it.
 */
export function StatusBadge({
  status = "neutral",
  children,
  className,
}: StatusBadgeProps): JSX.Element {
  const classes = [
    "drv-status-badge",
    `drv-status-badge-${status}`,
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return <span className={classes}>{children}</span>;
}
