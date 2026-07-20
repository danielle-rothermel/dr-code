import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  CodeBlock: ({ code }: { code: string }) => <pre>{code}</pre>,
  CodeDiff: () => <div>source diff</div>,
  StatusBadge: ({ children }: { children: string | number }) => <span>{children}</span>,
}));

import { Analysis } from "../src/analysis";
import { evaluatedData } from "./evaluation-fixture";

afterEach(cleanup);

describe("Analysis", () => {
  it("filters spot checks by source metadata and exposes the matching trace", () => {
    render(<Analysis />);

    fireEvent.change(screen.getByLabelText("Search examples"), {
      target: { value: "legacy_domain_partial" },
    });

    expect(screen.getByText("No decoder response was present in this sample.")).toBeTruthy();
    expect(screen.getByText("legacy_domain_partial")).toBeTruthy();
    expect(screen.getAllByText(/decoder output missing/i).length).toBeGreaterThan(0);
  });

  it("updates crosstab data when the selected outcome changes", () => {
    render(<Analysis />);

    fireEvent.change(screen.getByLabelText("Crosstab outcome", { selector: "select" }), {
      target: { value: "decoder_output_missing" },
    });

    expect(screen.getByRole("columnheader", { name: "Outcome count" })).toBeTruthy();
    expect(screen.getByText("legacy_domain_partial")).toBeTruthy();
  });

  it("keeps evaluation conclusions explicit when joined data is absent", () => {
    const preprocessingOnlyData = { ...evaluatedData };
    delete preprocessingOnlyData.candidate_evaluation;

    render(<Analysis data={preprocessingOnlyData} />);

    expect(screen.getByRole("heading", {
      name: "Candidate execution results are not part of this snapshot yet.",
    })).toBeTruthy();
  });

  it("keeps snapshots without the optional failure browser usable", () => {
    const compatibleData = { ...evaluatedData };
    delete compatibleData.failure_browser;

    render(<Analysis data={compatibleData} />);

    expect(screen.queryByRole("link", { name: "Failures" })).toBeNull();
    expect(screen.queryByRole("heading", { name: /Inspect every nonblank response/i })).toBeNull();
    expect(screen.getByRole("heading", { name: /Twelve diagnostic examples/i })).toBeTruthy();
  });

  it("renders the candidate funnel and sample best-of findings when evaluation is present", () => {
    render(<Analysis data={evaluatedData} />);

    expect(screen.getByRole("heading", { name: /Execution decides whether the code works/i })).toBeTruthy();
    expect(screen.getByText("Candidate pass rate")).toBeTruthy();
    expect(screen.getByText("Sample best-of pass rate")).toBeTruthy();
    expect(screen.getByText("evaluation attempted candidates")).toBeTruthy();
    expect(screen.getByText("Individual candidate test results")).toBeTruthy();
  });

  it("surfaces manifest-backed evaluation context and limitations", () => {
    render(<Analysis data={evaluatedData} />);

    expect(screen.getByText("fixture@1")).toBeTruthy();
    expect(screen.getByText("code_test@1")).toBeTruthy();
    expect(screen.getByText("fixture-runner@1")).toBeTruthy();
    expect(screen.getByText("0123456789ab…89abcdef")).toBeTruthy();
    expect(screen.getByTitle("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")).toBeTruthy();
    expect(screen.getByText("Fixture evaluation results apply only to the supplied HumanEval+ profile.")).toBeTruthy();
  });

  it("keeps earlier evaluation payloads usable without provenance", () => {
    const evaluation = evaluatedData.candidate_evaluation;
    if (!evaluation) throw new Error("evaluated fixture is missing candidate evaluation");
    const compatibleData = {
      ...evaluatedData,
      candidate_evaluation: {
        ...evaluation,
        summary: {
          ...evaluation.summary,
          limitations: undefined,
          provenance: undefined,
        },
      },
    };

    render(<Analysis data={compatibleData} />);

    expect(screen.queryByLabelText("Evaluation provenance and limitations")).toBeNull();
    expect(screen.getByText("Candidate pass rate")).toBeTruthy();
  });

  it("filters infrastructure examples by diagnostic evidence", () => {
    render(<Analysis data={evaluatedData} />);

    fireEvent.change(screen.getByLabelText("Search test examples"), {
      target: { value: "SandboxError" },
    });

    expect(screen.getByText("runtime unavailable")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "candidate-infrastructure_failure" })).toBeTruthy();
    expect(screen.queryByText("candidate-passed")).toBeNull();
  });

  it("switches the evaluation comparison to a source dimension", () => {
    render(<Analysis data={evaluatedData} />);

    expect(screen.getByText("Counts are final-candidate origin attributions; pass rate uses attributed candidates.")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Comparison lens"), {
      target: { value: "multiplicity" },
    });
    expect(screen.getByText(/Multiplicity rows include only samples with an extracted candidate/)).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Comparison lens"), {
      target: { value: "dimension:source_kind" },
    });

    expect(screen.getByText(/Pass \/ all includes samples with no extracted candidate/)).toBeTruthy();
    expect(screen.getByRole("cell", { name: "alpha" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "beta" })).toBeTruthy();
  });
});
