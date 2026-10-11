// Runs the connectome classifier off the main thread.
import { classify } from "./linalg.js";

self.onmessage = (event) => {
  const { ts, T, R, params, bias, windows } = event.data;
  try {
    const result = classify(ts, T, R, { params, bias, windows }, (done, total) => self.postMessage({ progress: done / total }));
    self.postMessage({ result });
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};
