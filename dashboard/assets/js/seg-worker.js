// The segmentation network in its own thread. A window of 160 x 128 x 128 voxels takes about a gigabyte
// and, on the CPU, close to a minute in WebAssembly; on the page's thread that would freeze the tab.

let ort = null;
let session = null;

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "init") {
      ort = await import(`${msg.ortUrl}${msg.gpu ? "ort.webgpu.min.mjs" : "ort.min.mjs"}`);
      ort.env.wasm.wasmPaths = msg.ortUrl;
      // more threads need a cross-origin isolated page, which GitHub Pages does not serve
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(msg.model, {
        executionProviders: msg.gpu ? ["webgpu", "wasm"] : ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({ type: "ready" });
    } else if (msg.type === "run") {
      const res = await session.run({ image: new ort.Tensor("float32", msg.tile, msg.dims) });
      const logits = new Float32Array(res.logits.data);
      res.logits.dispose?.();
      self.postMessage({ type: "logits", logits }, [logits.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
};
