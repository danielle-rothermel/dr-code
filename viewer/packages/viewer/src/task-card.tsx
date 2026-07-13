import type { JSX } from "react";

import { CodeBlock } from "./code-block.js";
import type { HumanEvalTask } from "./types.js";

export interface TaskCardProps {
  task: HumanEvalTask;
  className?: string;
}

/**
 * Server-tier card for a HumanEval task: prompt, canonical solution,
 * and test, all as zero-JS code panels.
 *
 * Panels are awaited directly (not composed as async JSX children) so
 * the returned tree contains only plain elements — renderable by
 * client react-dom too, which keeps the component testable without an
 * RSC runtime.
 */
export async function TaskCard({ task, className }: TaskCardProps): Promise<JSX.Element> {
  const classes = className ? `drv-task-card ${className}` : "drv-task-card";
  const [prompt, solution, test] = await Promise.all([
    CodeBlock({ code: task.prompt }),
    CodeBlock({ code: task.canonical_solution }),
    CodeBlock({ code: task.test }),
  ]);
  return (
    <section className={classes}>
      <header className="drv-task-header">
        <h3>{task.task_id}</h3>
        <code>{task.entry_point}</code>
      </header>
      {task.notes != null && task.notes.length > 0 && (
        <ul className="drv-task-notes">
          {task.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      <h4>Prompt</h4>
      {prompt}
      <h4>Canonical solution</h4>
      {solution}
      <h4>Test</h4>
      {test}
    </section>
  );
}
