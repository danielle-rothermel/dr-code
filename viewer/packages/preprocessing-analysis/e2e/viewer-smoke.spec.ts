import { expect, test } from "@playwright/test";

test("slash IDs, annotation lifecycle, and guarded navigation work live", async ({ page }) => {
  let annotationPuts = 0;
  let observeTargetPut!: () => void;
  const targetPutObserved = new Promise<void>((resolve) => {
    observeTargetPut = resolve;
  });
  let releaseTargetPutRequest!: () => void;
  const releaseTargetPut = new Promise<void>((resolve) => {
    releaseTargetPutRequest = resolve;
  });
  await page.route("**/api/annotations/**", async (route) => {
    if (route.request().method() === "PUT") {
      annotationPuts += 1;
      const body: unknown = route.request().postDataJSON();
      if (
        typeof body === "object"
        && body !== null
        && "note" in body
        && body.note === "flush before navigation"
      ) {
        observeTargetPut();
        await releaseTargetPut;
      }
    }
    await route.continue();
  });

  await page.goto("/");
  await page.getByRole("button", {
    name: /Inspect 19 examples at Corpus rows/,
  }).click();
  await expect(
    page.getByRole("heading", { name: "HumanEval/32" }),
  ).toBeVisible();

  await page.getByRole("button", { exact: true, name: "Review" }).click();
  const card = page.getByRole("article", { name: "Example HumanEval/32" });
  await expect(card).toBeVisible();
  await card.getByLabel("Comment").fill("browser annotation");
  await card.getByLabel("Comment").blur();
  await expect(card.getByText("Saved")).toBeVisible();

  const annotationDeleted = page.waitForResponse((response) => (
    response.request().method() === "DELETE"
    && response.url().includes("/api/annotations/")
    && response.ok()
  ));
  await card.getByRole("button", { name: "Delete annotation" }).click();
  await annotationDeleted;
  await expect(card.getByLabel("Comment")).toHaveValue("");
  await expect(
    card.getByRole("button", { name: "Delete annotation" }),
  ).toBeDisabled();

  await card.getByLabel("Comment").fill("flush before navigation");
  let pageTransitions = 0;
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      url.pathname.endsWith("/api/review-examples")
      && url.searchParams.get("offset") !== "0"
    ) {
      pageTransitions += 1;
    }
  });
  try {
    await page.evaluate(() => {
      const next = [...document.querySelectorAll("button")].find(
        (button) => button.textContent === "Next page",
      );
      if (!(next instanceof HTMLButtonElement)) {
        throw new Error("Next page button is missing");
      }
      next.click();
    });
    await targetPutObserved;
    await expect(page.getByText(/Page 1 of 2 · 11 examples/)).toBeVisible();
    await expect(page.getByLabel("Review controls")).toHaveAttribute("aria-busy", "true");
    await expect(page.getByRole("button", { name: "Next page" })).toBeDisabled();

    await page.evaluate(() => {
      const next = [...document.querySelectorAll("button")].find(
        (button) => button.textContent === "Next page",
      );
      if (!(next instanceof HTMLButtonElement)) {
        throw new Error("Next page button is missing");
      }
      next.disabled = false;
      next.click();
    });
  } finally {
    releaseTargetPutRequest();
  }
  await expect(page.getByText(/Page 2 of 2 · 11 examples/)).toBeVisible();
  expect(pageTransitions).toBe(1);
  expect(annotationPuts).toBe(2);
});

test("deleting an annotation cancels failed tag creation before navigation", async ({ page }) => {
  let annotationPuts = 0;
  let tagPosts = 0;
  await page.route("**/api/annotations/**", async (route) => {
    if (route.request().method() === "PUT") annotationPuts += 1;
    await route.continue();
  });
  await page.route("**/api/tags", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    tagPosts += 1;
    await route.fulfill({
      body: JSON.stringify({ detail: "tag database is locked" }),
      contentType: "application/json",
      status: 500,
    });
  });

  await page.goto("/");
  await page.getByRole("button", {
    name: /Inspect 19 examples at Corpus rows/,
  }).click();
  await page.getByRole("button", { exact: true, name: "Review" }).click();
  const card = page.getByRole("article", { name: "Example HumanEval/32" });
  await expect(card).toBeVisible();

  await card.getByRole("radio", { name: "Flag" }).click();
  await expect(card.getByText("Saved")).toBeVisible();
  await card.getByLabel("Create tag").fill("retry tag");
  await card.getByRole("button", { name: "Create and select" }).click();
  await expect(card.getByText("Tag save failed")).toBeVisible();

  const deleted = page.waitForRequest((request) => (
    request.method() === "DELETE" && request.url().includes("/api/annotations/")
  ));
  await card.getByRole("button", { name: "Delete annotation" }).click();
  await deleted;
  await expect(card.getByRole("button", { name: "Delete annotation" })).toBeDisabled();

  await page.getByRole("button", { name: "Next page" }).click();
  await expect(page.getByText(/Page 2 of 2 · 11 examples/)).toBeVisible();
  expect(tagPosts).toBe(1);
  expect(annotationPuts).toBe(1);

  const exported = await page.request.get("/api/annotations/export");
  expect(exported.ok()).toBe(true);
  expect(await exported.json()).toEqual([]);
});
