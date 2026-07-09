// jsdom lacks ResizeObserver, which DiffView observes for wrapping.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

globalThis.ResizeObserver ??=
  ResizeObserverStub as unknown as typeof ResizeObserver;

// jsdom's canvas getContext returns null (no node-canvas); DiffView
// measures text width through a 2D context.
const canvasContextStub = {
  font: "",
  measureText: (text: string) => ({ width: text.length * 8 }),
};

HTMLCanvasElement.prototype.getContext = (() =>
  canvasContextStub) as unknown as typeof HTMLCanvasElement.prototype.getContext;
