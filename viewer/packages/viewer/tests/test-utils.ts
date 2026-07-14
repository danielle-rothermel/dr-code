/**
 * React's scheduler flushes each commit's passive-effect bookkeeping in its
 * own `setImmediate` tick (see the "scheduler" package's Node host config,
 * which falls back to `setImmediate` when there's no `MessageChannel`).
 * When a state update comes from a `useEffect`-driven promise (not a user
 * event), that commit is not wrapped in `act()`, so its trailing
 * `setImmediate` callback is scheduled independently of the assertion that
 * observes the resulting DOM change.
 *
 * `waitFor`/`findBy*` can resolve as soon as a MutationObserver sees the
 * DOM update, which can happen before that same commit's own
 * `setImmediate` callback has had a turn on the event loop. If the test
 * ends there, vitest tears down the jsdom environment before the pending
 * callback runs, and it later throws `ReferenceError: window is not
 * defined` from inside React's scheduler as an unhandled error.
 *
 * Awaiting one `setImmediate` tick after the target DOM state appears lets
 * any already-queued scheduler callback run first (Node drains
 * `setImmediate` callbacks in FIFO order) while the environment is still
 * alive.
 *
 * This package has no @types/node dependency (it's browser-facing), so
 * `setImmediate` is declared ambiently here rather than pulling in Node's
 * global types project-wide just for this one test helper.
 */
declare const setImmediate: (callback: () => void) => void;

export function flushScheduler(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}
