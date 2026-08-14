import {mountColmapViewer} from "./main";

export * from "./main";

const root = document.querySelector<HTMLElement>("#colmap-viewer-root");
if (root) {
  try {
    mountColmapViewer(root);
  } catch (error) {
    console.error("[COLMAP viewer] Initialization failed", error);
  }
}
