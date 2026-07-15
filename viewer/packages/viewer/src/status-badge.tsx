import type { JSX, ReactNode } from "react";

export type StatusBadgeStatus =
  | "success"
  | "failure"
  | "warning"
  | "neutral";

export interface StatusBadgeProps {
  status: StatusBadgeStatus;
  children: ReactNode;
  theme?: "light" | "dark";
  className?: string;
}

/** A compact label with a semantic status color. */
export function StatusBadge({
  status,
  children,
  theme = "light",
  className,
}: StatusBadgeProps): JSX.Element {
  const classes = className
    ? `drv-status-badge ${className}`
    : "drv-status-badge";
  return (
    <span className={classes} data-status={status} data-theme={theme}>
      {children}
    </span>
  );
}
