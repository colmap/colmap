import {CAMERA_MODEL_NAMES, project} from "./camera_models";
import {projectionCenter, transformPoint} from "./math";
import {discoverSparseModels, normalizePath} from "./parser";
import type {ImageRecord, LocalFile, Point3D, Reconstruction, SparseModelCandidate} from "./types";
import {ReconstructionViewer} from "./viewer";

const root = document.querySelector<HTMLElement>("#colmap-viewer-root");
if (root) initialize(root);

let observationImageObserver: IntersectionObserver | null = null;

function initialize(container: HTMLElement): void {
  container.innerHTML = `
    <section class="colmap-viewer" aria-label="COLMAP sparse reconstruction viewer">
      <header class="viewer-toolbar">
        <div class="viewer-brand"><strong>3D Viewer</strong><span id="viewer-stats">No model loaded</span></div>
        <div class="viewer-actions">
          <button type="button" id="viewer-open">Open folder</button>
          <select id="viewer-model" aria-label="Sparse model" hidden></select>
          <button type="button" id="viewer-reset" disabled>Reset view</button>
          <button type="button" id="viewer-clear" disabled>Clear selection</button>
        </div>
      </header>
      <div class="viewer-controls" aria-label="Render controls">
        <label>Projection <select id="viewer-projection"><option value="perspective">Perspective</option><option value="orthographic">Orthographic</option></select></label>
        <label>Point size <input id="viewer-point-size" type="range" min="0.5" max="100" step="0.5" value="2"></label>
        <label>Camera size <input id="viewer-camera-size" type="range" min="0.000001" max="1" step="0.0005" value="0.025"></label>
        <label>Min. track <input id="viewer-track" type="number" min="0" max="100" value="3"></label>
        <label>Max. error <input id="viewer-error" type="number" min="0" step="0.25" value="2"></label>
        <label class="viewer-check"><input id="viewer-connections" type="checkbox"> Connections</label>
      </div>
      <div class="viewer-workspace">
        <div class="viewer-stage">
          <canvas id="viewer-canvas" aria-label="Interactive 3D reconstruction"></canvas>
          <div id="viewer-drop" class="viewer-drop">
            <div class="viewer-drop-mark" aria-hidden="true">3D</div>
            <h2>Open a COLMAP reconstruction</h2>
            <p>Drop a workspace or sparse model folder here.</p>
            <p class="viewer-drop-detail">Requires <code>cameras.bin</code>, <code>images.bin</code>, and <code>points3D.bin</code>. Source images are optional and never leave this browser.</p>
            <button type="button" id="viewer-drop-open">Choose folder</button>
          </div>
          <div id="viewer-status" class="viewer-status" role="status" hidden></div>
        </div>
        <aside id="viewer-inspector" class="viewer-inspector" aria-live="polite">
          <div class="viewer-inspector-empty"><h2>Inspector</h2><p>Double-click a point or camera to inspect it.</p></div>
        </aside>
      </div>
      <input id="viewer-folder-input" type="file" multiple hidden>
    </section>`;

  const canvas = requireElement<HTMLCanvasElement>(container, "#viewer-canvas");
  let viewer: ReconstructionViewer;
  try {
    viewer = new ReconstructionViewer(canvas);
  } catch (error) {
    showFatal(container, `WebGL2 is unavailable: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }

  const drop = requireElement<HTMLElement>(container, "#viewer-drop");
  const status = requireElement<HTMLElement>(container, "#viewer-status");
  const inspector = requireElement<HTMLElement>(container, "#viewer-inspector");
  const stats = requireElement<HTMLElement>(container, "#viewer-stats");
  const modelSelect = requireElement<HTMLSelectElement>(container, "#viewer-model");
  const input = requireElement<HTMLInputElement>(container, "#viewer-folder-input");
  input.setAttribute("webkitdirectory", "");
  let allEntries: LocalFile[] = [];
  let candidates: SparseModelCandidate[] = [];
  let reconstruction: Reconstruction | null = null;
  let imageFiles = new Map<string, File>();
  let currentSelection: {type: "point"; value: Point3D} | {type: "image"; value: ImageRecord} | null = null;
  let activeLoad: AbortController | null = null;

  const setStatus = (message: string | null, error = false): void => {
    status.hidden = message === null;
    status.textContent = message ?? "";
    status.title = message ?? "";
    status.classList.toggle("is-error", error);
  };

  const refreshImages = (): void => {
    imageFiles = new Map(allEntries.map((entry) => [normalizePath(entry.path), entry.file]));
    if (currentSelection && reconstruction) renderInspector(inspector, currentSelection, reconstruction, imageFiles);
  };

  const clearLoadedModel = (): void => {
    reconstruction = null;
    currentSelection = null;
    viewer.clearReconstruction();
    stats.textContent = "No model loaded";
    for (const button of container.querySelectorAll<HTMLButtonElement>("#viewer-reset, #viewer-clear")) button.disabled = true;
    inspector.innerHTML = `<div class="viewer-inspector-empty"><h2>Inspector</h2><p>Double-click a point or camera to inspect it.</p></div>`;
  };

  const loadCandidate = async (candidate: SparseModelCandidate): Promise<void> => {
    activeLoad?.abort();
    const load = new AbortController();
    activeLoad = load;
    clearLoadedModel();
    setStatus("Parsing reconstruction...");
    let phase = "parse model";
    try {
      const parsed = await parseInWorker(candidate.files, load.signal);
      if (load.signal.aborted || activeLoad !== load) return;
      phase = "build Three.js scene";
      viewer.setReconstruction(parsed);
      if (load.signal.aborted || activeLoad !== load) return;
      reconstruction = parsed;
      drop.hidden = true;
      setStatus(null);
      stats.textContent = `${reconstruction.images.size.toLocaleString()} cameras / ${viewer.visiblePointCount.toLocaleString()} visible points`;
      for (const button of container.querySelectorAll<HTMLButtonElement>("#viewer-reset, #viewer-clear")) button.disabled = false;
      currentSelection = null;
      inspector.innerHTML = `<div class="viewer-inspector-empty"><h2>Model loaded</h2><p>Double-click a point or camera to inspect it.</p><dl><dt>Format</dt><dd>${reconstruction.modernRigFormat ? "Binary with rigs" : "Legacy binary"}</dd><dt>Images</dt><dd>${reconstruction.images.size.toLocaleString()}</dd><dt>Points</dt><dd>${reconstruction.points3D.size.toLocaleString()}</dd></dl></div>`;
      refreshImages();
    } catch (error) {
      if (load.signal.aborted || activeLoad !== load || (error instanceof DOMException && error.name === "AbortError")) return;
      clearLoadedModel();
      drop.hidden = false;
      const detail = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      console.error(`[COLMAP viewer] Failed to ${phase}`, error);
      setStatus(`Failed to ${phase}: ${detail || "Unknown error"}`, true);
    } finally {
      if (activeLoad === load) activeLoad = null;
    }
  };

  const acceptEntries = async (entries: LocalFile[]): Promise<void> => {
    if (entries.length === 0) return;
    const found = discoverSparseModels(entries);
    if (found.length === 0 && reconstruction) {
      allEntries.push(...entries);
      refreshImages();
      setStatus(`Added ${entries.length.toLocaleString()} image files`);
      window.setTimeout(() => setStatus(null), 1800);
      return;
    }
    if (found.length === 0) {
      setStatus("No binary sparse model was found in that folder", true);
      return;
    }
    allEntries = entries;
    candidates = found;
    refreshImages();
    modelSelect.replaceChildren(...candidates.map((candidate, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = candidate.path === "." ? "Sparse model" : candidate.path;
      return option;
    }));
    modelSelect.hidden = candidates.length < 2;
    await loadCandidate(candidates[0]!);
  };

  const openPicker = (): void => input.click();
  requireElement(container, "#viewer-open").addEventListener("click", openPicker);
  requireElement(container, "#viewer-drop-open").addEventListener("click", openPicker);
  input.addEventListener("change", async () => {
    const entries = [...(input.files ?? [])].map((file) => ({path: file.webkitRelativePath || file.name, file}));
    input.value = "";
    await acceptEntries(entries);
  });
  modelSelect.addEventListener("change", () => {
    const candidate = candidates[Number(modelSelect.value)];
    if (candidate) void loadCandidate(candidate);
  });

  for (const eventName of ["dragenter", "dragover"] as const) container.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.hidden = false;
    drop.classList.add("is-dragging");
  });
  container.addEventListener("dragleave", (event) => {
    if (!container.contains(event.relatedTarget as Node | null)) drop.classList.remove("is-dragging");
  });
  container.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("is-dragging");
    const entries = await filesFromDrop(event.dataTransfer);
    await acceptEntries(entries);
    if (reconstruction) drop.hidden = true;
  });

  viewer.onSelection = (selection) => {
    currentSelection = selection;
    if (selection && reconstruction) renderInspector(inspector, selection, reconstruction, imageFiles);
    else inspector.innerHTML = `<div class="viewer-inspector-empty"><h2>Inspector</h2><p>Double-click a point or camera to inspect it.</p></div>`;
  };
  viewer.onError = (error) => {
    console.error("[COLMAP viewer] WebGL render failed", error);
    setStatus(`WebGL render failed: ${error.name}: ${error.message || "Unknown error"}`, true);
  };
  viewer.onSettingsChange = (settings) => {
    requireElement<HTMLInputElement>(container, "#viewer-point-size").value = String(settings.pointSize);
    requireElement<HTMLInputElement>(container, "#viewer-camera-size").value = String(settings.cameraSize);
  };
  requireElement(container, "#viewer-reset").addEventListener("click", () => viewer.resetView());
  requireElement(container, "#viewer-clear").addEventListener("click", () => viewer.clearSelection());
  bindControls(container, viewer);
}

function requireElement<T extends Element = HTMLElement>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing viewer element ${selector}`);
  return element;
}

function bindControls(root: ParentNode, viewer: ReconstructionViewer): void {
  requireElement<HTMLSelectElement>(root, "#viewer-projection").addEventListener("change", (event) => viewer.updateSettings({projection: (event.currentTarget as HTMLSelectElement).value as "perspective" | "orthographic"}));
  requireElement<HTMLInputElement>(root, "#viewer-point-size").addEventListener("input", (event) => viewer.updateSettings({pointSize: Number((event.currentTarget as HTMLInputElement).value)}));
  requireElement<HTMLInputElement>(root, "#viewer-camera-size").addEventListener("input", (event) => viewer.updateSettings({cameraSize: Number((event.currentTarget as HTMLInputElement).value)}));
  requireElement<HTMLInputElement>(root, "#viewer-track").addEventListener("change", (event) => viewer.updateSettings({minTrackLength: Math.max(0, Number((event.currentTarget as HTMLInputElement).value))}));
  requireElement<HTMLInputElement>(root, "#viewer-error").addEventListener("change", (event) => viewer.updateSettings({maxError: Math.max(0, Number((event.currentTarget as HTMLInputElement).value))}));
  requireElement<HTMLInputElement>(root, "#viewer-connections").addEventListener("change", (event) => viewer.updateSettings({showConnections: (event.currentTarget as HTMLInputElement).checked}));
}

async function parseInWorker(files: Map<string, File>, signal: AbortSignal): Promise<Reconstruction> {
  const worker = new Worker(new URL("./parser.worker.ts", import.meta.url), {type: "module"});
  return await new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback: () => void): void => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", abort);
      worker.terminate();
      callback();
    };
    const abort = (): void => finish(() => reject(new DOMException("Model load superseded", "AbortError")));
    if (signal.aborted) {
      abort();
      return;
    }
    signal.addEventListener("abort", abort, {once: true});
    worker.onmessage = (event: MessageEvent<{ok: boolean; reconstruction?: Reconstruction; error?: string}>) => {
      if (event.data.ok && event.data.reconstruction) finish(() => resolve(event.data.reconstruction!));
      else finish(() => reject(new Error(event.data.error ?? "Could not parse reconstruction")));
    };
    worker.onerror = (event) => finish(() => reject(new Error(event.message || "Parser worker failed")));
    worker.onmessageerror = () => finish(() => reject(new Error("Parser worker returned an unreadable result")));
    worker.postMessage(files);
  });
}

async function filesFromDrop(transfer: DataTransfer | null): Promise<LocalFile[]> {
  if (!transfer) return [];
  const roots = [...transfer.items].map((item) => item.webkitGetAsEntry()).filter((entry): entry is FileSystemEntry => entry !== null);
  if (roots.length > 0) {
    const files = await Promise.all(roots.map((entry) => readEntry(entry, entry.name)));
    return files.flat();
  }
  return [...transfer.files].map((file) => ({path: file.name, file}));
}

async function readEntry(entry: FileSystemEntry, path: string): Promise<LocalFile[]> {
  if (entry.isFile) return [{path, file: await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject))}];
  if (!entry.isDirectory) return [];
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const children: FileSystemEntry[] = [];
  while (true) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (batch.length === 0) break;
    children.push(...batch);
  }
  const nested = await Promise.all(children.map((child) => readEntry(child, `${path}/${child.name}`)));
  return nested.flat();
}

function findImageFile(files: Map<string, File>, name: string): File | null {
  const normalized = normalizePath(name);
  const direct = files.get(normalized);
  if (direct) return direct;
  const suffix = `/${normalized}`;
  let best: [string, File] | null = null;
  for (const entry of files) if (entry[0].endsWith(suffix) && (!best || entry[0].length < best[0].length)) best = entry;
  return best?.[1] ?? null;
}

function metadata(title: string, rows: Array<[string, string]>): HTMLElement {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const list = document.createElement("dl");
  for (const [key, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = key;
    const description = document.createElement("dd");
    description.textContent = value;
    list.append(term, description);
  }
  section.append(heading, list);
  return section;
}

function renderInspector(
  inspector: HTMLElement,
  selection: {type: "point"; value: Point3D} | {type: "image"; value: ImageRecord},
  reconstruction: Reconstruction,
  files: Map<string, File>,
): void {
  observationImageObserver?.disconnect();
  observationImageObserver = null;
  inspector.replaceChildren();
  if (selection.type === "image") {
    const image = selection.value;
    const camera = reconstruction.cameras.get(image.cameraId)!;
    const triangulated = image.points2D.filter((point) => point.point3DId !== null).length;
    const center = projectionCenter(image.camFromWorld);
    inspector.append(metadata(`Image ${image.id}`, [
      ["Image", image.name],
      ["Camera model", CAMERA_MODEL_NAMES[camera.modelId] ?? `Unknown (${camera.modelId})`],
      ["Dimensions", `${camera.width} x ${camera.height}`],
      ["Frame / rig", `${image.frameId} / ${image.rigId}`],
      ["Observations", `${triangulated.toLocaleString()} / ${image.points2D.length.toLocaleString()} triangulated`],
      ["Center", center.map((value) => value.toFixed(6)).join(", ")],
      ["Pose (qw,qx,qy,qz | tx,ty,tz)", `${image.camFromWorld.rotation.map((value) => value.toFixed(6)).join(", ")} | ${image.camFromWorld.translation.map((value) => value.toFixed(6)).join(", ")}`],
    ]));
    const file = findImageFile(files, image.name);
    const figure = document.createElement("figure");
    figure.className = "viewer-image";
    if (file) {
      const canvas = document.createElement("canvas");
      figure.append(canvas);
      void drawCameraImage(canvas, file, image, camera.width, camera.height).catch(() => showMissingImage(figure, "The browser could not decode this image."));
    } else showMissingImage(figure, "Source image unavailable. Drop the images folder to display keypoints.");
    inspector.append(figure);
  } else {
    const point = selection.value;
    inspector.append(metadata(`Point ${point.id}`, [
      ["Position", point.xyz.map((value) => value.toFixed(8)).join(", ")],
      ["Color", point.color.join(", ")],
      ["Error", `${point.error.toFixed(4)} px`],
      ["Track length", point.track.length.toLocaleString()],
    ]));
    const heading = document.createElement("h3");
    heading.textContent = "Observations";
    const gallery = document.createElement("div");
    gallery.className = "viewer-observations";
    const tracks = [...point.track].sort((a, b) => (reconstruction.images.get(a.imageId)?.name ?? "").localeCompare(reconstruction.images.get(b.imageId)?.name ?? ""));
    for (const track of tracks) {
      const image = reconstruction.images.get(track.imageId);
      if (!image) continue;
      const observed = image.points2D[track.point2DIdx];
      const camera = reconstruction.cameras.get(image.cameraId);
      if (!observed || !camera) continue;
      const projected = project(camera, transformPoint(image.camFromWorld, point.xyz));
      const card = document.createElement("article");
      card.className = "viewer-observation";
      const label = document.createElement("div");
      label.className = "viewer-observation-label";
      label.textContent = image.name;
      const detail = document.createElement("small");
      detail.textContent = `Image ${image.id}`;
      label.append(detail);
      card.append(label);
      const file = findImageFile(files, image.name);
      if (file) {
        const canvas = document.createElement("canvas");
        canvas.width = 280;
        canvas.height = 180;
        card.prepend(canvas);
        if (projected) {
          const error = Math.hypot(observed.xy[0] - projected[0], observed.xy[1] - projected[1]);
          detail.textContent = `Image ${image.id} / ${error.toFixed(3)} px`;
        }
        const load = (): void => {
          void drawObservation(canvas, file, observed.xy, projected).catch(() => { canvas.replaceWith(document.createTextNode("Image unavailable")); });
        };
        if ("IntersectionObserver" in window) {
          observationImageObserver ??= new IntersectionObserver((entries, observer) => {
            for (const entry of entries) {
              if (!entry.isIntersecting) continue;
              observer.unobserve(entry.target);
              (entry.target as HTMLElement).dispatchEvent(new Event("viewer-load-image"));
            }
          }, {root: inspector, rootMargin: "180px"});
          canvas.addEventListener("viewer-load-image", load, {once: true});
          observationImageObserver.observe(canvas);
        } else load();
      } else {
        const missing = document.createElement("div");
        missing.className = "viewer-observation-missing";
        missing.textContent = "Image unavailable";
        card.prepend(missing);
      }
      gallery.append(card);
    }
    inspector.append(heading, gallery);
  }
}

async function drawCameraImage(canvas: HTMLCanvasElement, file: File, image: ImageRecord, width: number, height: number): Promise<void> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1000 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const sx = canvas.width / width;
  const sy = canvas.height / height;
  for (const point of image.points2D) {
    context.fillStyle = point.point3DId === null ? "#ef3028" : "#ff00ff";
    context.beginPath();
    context.arc(point.xy[0] * sx, point.xy[1] * sy, 1.6, 0, Math.PI * 2);
    context.fill();
  }
}

async function drawObservation(canvas: HTMLCanvasElement, file: File, observed: [number, number], projected: [number, number] | null): Promise<void> {
  const bitmap = await createImageBitmap(file);
  const centerX = projected ? (observed[0] + projected[0]) / 2 : observed[0];
  const centerY = projected ? (observed[1] + projected[1]) / 2 : observed[1];
  const distance = projected ? Math.hypot(observed[0] - projected[0], observed[1] - projected[1]) : 0;
  const crop = Math.min(Math.max(120, distance * 2 + 50), Math.min(bitmap.width, bitmap.height));
  const sourceX = Math.max(0, Math.min(bitmap.width - crop, centerX - crop / 2));
  const sourceY = Math.max(0, Math.min(bitmap.height - crop, centerY - crop / 2));
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, sourceX, sourceY, crop, crop, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const mapPoint = (xy: [number, number]): [number, number] => [(xy[0] - sourceX) / crop * canvas.width, (xy[1] - sourceY) / crop * canvas.height];
  const [ox, oy] = mapPoint(observed);
  context.strokeStyle = "#00e13a";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(ox - 9, oy - 9); context.lineTo(ox + 9, oy + 9);
  context.moveTo(ox - 9, oy + 9); context.lineTo(ox + 9, oy - 9);
  context.stroke();
  if (projected) {
    const [px, py] = mapPoint(projected);
    context.strokeStyle = "#ef3028";
    context.lineWidth = 2;
    for (const radius of [4, 12, 30]) { context.beginPath(); context.arc(px, py, radius, 0, Math.PI * 2); context.stroke(); }
  }
}

function showMissingImage(figure: HTMLElement, message: string): void {
  figure.replaceChildren();
  const note = document.createElement("p");
  note.className = "viewer-image-missing";
  note.textContent = message;
  figure.append(note);
}

function showFatal(container: HTMLElement, message: string): void {
  container.replaceChildren();
  const alert = document.createElement("div");
  alert.className = "admonition error";
  alert.setAttribute("role", "alert");
  alert.textContent = message;
  container.append(alert);
}
