import {readdir, readFile} from "node:fs/promises";

import {expect, test} from "vitest";

test("uses a viewer-relative parser worker URL", async () => {
  const names = await readdir("_static/viewer", {recursive: true});
  const scripts = names.filter((name) => name.endsWith(".js"));
  const bundle = (await Promise.all(scripts.map((name) => readFile(`_static/viewer/${name}`, "utf8")))).join("\n");
  expect(bundle).toMatch(/new URL\("assets\/parser\.worker-[A-Za-z0-9_-]+\.js"/);
  expect(bundle).not.toMatch(/new URL\("\/assets\/parser\.worker-/);
});

test("emits an independently reusable component and scoped stylesheet", async () => {
  const component = await readFile("_static/viewer/component.js", "utf8");
  const styles = await readFile("_static/viewer/viewer.css", "utf8");
  expect(component).toContain("mountColmapViewer");
  expect(component).not.toContain("#colmap-viewer-root");
  expect(styles).toContain(".colmap-viewer-host .viewer-workspace");
  expect(styles).not.toMatch(/(^|})\.viewer-workspace/);
});
