import {parseReconstruction} from "./parser";

self.onmessage = async (event: MessageEvent<Map<string, File>>) => {
  try {
    const reconstruction = await parseReconstruction(event.data);
    self.postMessage({ok: true, reconstruction});
  } catch (error) {
    self.postMessage({ok: false, error: error instanceof Error ? error.message : String(error)});
  }
};
