const numberFormatter = new Intl.NumberFormat("en-US");
const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
  style: "percent",
});

export function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

export function formatPercent(value: number | null): string {
  return value === null ? "—" : percentFormatter.format(value);
}

export function formatDelta(value: number): string {
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

export function formatRateDelta(value: number | null): string {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${percentFormatter.format(value)}`;
}

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unknown error occurred";
}
