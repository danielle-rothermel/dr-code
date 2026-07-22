import type { CandidateOrigin, JsonValue } from "./api";
import { humanize } from "./format";

function formatDetail(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function CandidateOrigins({ origins }: { origins: CandidateOrigin[] }) {
  if (origins.length === 0) return null;

  return (
    <section className="candidate-origins" aria-label="Extraction origins">
      <p>
        {origins.length === 1
          ? "1 extraction path"
          : `${origins.length} converged extraction paths`}
      </p>
      <ol className="candidate-origin-list">
        {origins.map((origin, originIndex) => (
          <li key={originIndex}>
            <span className="candidate-origin-label">Path {originIndex + 1}</span>
            {origin.path.length === 0 ? (
              <span className="candidate-origin-empty">No extraction operations recorded</span>
            ) : (
              <ol className="candidate-origin-path">
                {origin.path.map((operation, operationIndex) => (
                  <li key={`${operation.kind}-${operationIndex}`}>
                    <strong>{humanize(operation.kind)}</strong>
                    {Object.keys(operation.details).length > 0 && (
                      <dl>
                        {Object.entries(operation.details).map(([key, value]) => (
                          <div key={key}>
                            <dt>{humanize(key)}</dt>
                            <dd>{formatDetail(value)}</dd>
                          </div>
                        ))}
                      </dl>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
