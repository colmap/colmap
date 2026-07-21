import {readFile} from "node:fs/promises";

import {expect, test} from "vitest";

test("uses a viewer-relative parser worker URL", async () => {
  let bundle: string;
  try {
    bundle = await readFile("_static/viewer/viewer.js", "utf8");
  } catch {
    return;
  }
  expect(bundle).toMatch(/new URL\("assets\/parser\.worker-[A-Za-z0-9_-]+\.js"/);
  expect(bundle).not.toMatch(/new URL\("\/assets\/parser\.worker-/);
});
