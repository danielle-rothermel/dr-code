import type { JSX } from "react";

import type { EvaluationCaseSummary } from "./types.js";

export interface EvaluationCaseTableProps {
  cases: EvaluationCaseSummary[];
  className?: string;
}

/**
 * Per-test-case evaluation results. Shared-tier: no client JS of its
 * own, renders in both server and client trees.
 */
export function EvaluationCaseTable({
  cases,
  className,
}: EvaluationCaseTableProps): JSX.Element {
  const classes = className ? `drv-case-table ${className}` : "drv-case-table";
  return (
    <table className={classes}>
      <thead>
        <tr>
          <th>case</th>
          <th>function</th>
          <th>status</th>
          <th>kind</th>
          <th>input</th>
          <th>expected</th>
          <th>actual</th>
          <th>message</th>
        </tr>
      </thead>
      <tbody>
        {cases.map((caseSummary) => (
          <tr key={caseSummary.case_id}>
            <td>{caseSummary.case_id}</td>
            <td>
              <code>{caseSummary.function_name}</code>
            </td>
            <td>
              <span
                className={`drv-case-status drv-case-status-${caseSummary.status}`}
              >
                {caseSummary.status}
              </span>
            </td>
            <td>{caseSummary.test_type}</td>
            <td>
              <code>{caseSummary.input_repr}</code>
            </td>
            <td>
              <code>{caseSummary.expected_output_repr}</code>
            </td>
            <td>
              <code>{caseSummary.actual_output_repr}</code>
            </td>
            <td>{caseSummary.message}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
