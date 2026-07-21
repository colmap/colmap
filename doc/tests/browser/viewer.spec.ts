import {expect, test} from "@playwright/test";

import {cameraFile, imageFile, pointFile} from "../binary_fixture";

test("initializes the local-only viewer shell", async ({page}) => {
  await page.goto("/tests/viewer.html");
  await expect(page.getByRole("heading", {name: "Open a COLMAP reconstruction"})).toBeVisible();
  await expect(page.getByRole("button", {name: "Choose folder"})).toBeVisible();
  await expect(page.locator('[data-viewer="canvas"]')).toBeAttached();
  await expect(page.locator('[data-viewer="title"]')).toHaveText("COLMAP - 3D Web Viewer");
  await expect(page.locator('[data-viewer="stats"]')).toBeHidden();

  const files = await Promise.all([cameraFile(), imageFile(), pointFile()].map(async (file) => ({
    name: file.name,
    mimeType: "application/octet-stream",
    buffer: Buffer.from(await file.arrayBuffer()),
  })));
  const input = page.locator('[data-viewer="folder-input"]');
  await input.evaluate((element) => element.removeAttribute("webkitdirectory"));
  await input.setInputFiles(files);
  await expect(page.locator('[data-viewer="stats"]')).toHaveText("1 cameras / 1 points");
  await expect(page.getByRole("heading", {name: "Model loaded"})).toBeVisible();
  await page.locator('[data-viewer="projection"]').selectOption("orthographic");
  await expect(page.locator('[data-viewer="projection"]')).toHaveValue("orthographic");

  const malformedFiles = files.map((file) => file.name === "cameras.bin" ? {...file, buffer: Buffer.from([1])} : file);
  await input.setInputFiles(malformedFiles);
  await expect(page.locator('[data-viewer="stats"]')).toBeHidden();
  await expect(page.getByRole("heading", {name: "Open a COLMAP reconstruction"})).toBeVisible();
  await expect(page.locator('[data-viewer="status"]')).toContainText("Failed to parse model");
  await expect(page.locator('[data-viewer="reset"]')).toBeDisabled();

  const lifecycle = await page.evaluate(async () => {
    const modulePath = "/viewer_src/main.ts";
    const {mountColmapViewer} = await import(/* @vite-ignore */ modulePath) as typeof import("../../viewer_src/main");
    const host = document.createElement("div");
    document.body.append(host);
    const embedded = mountColmapViewer(host, {title: "Embedded viewer"});
    const title = host.querySelector('[data-viewer="title"]')?.textContent;
    embedded.clear();
    embedded.dispose();
    return {title, childCount: host.childElementCount, hasHostClass: host.classList.contains("colmap-viewer-host")};
  });
  expect(lifecycle).toEqual({title: "Embedded viewer", childCount: 0, hasHostClass: false});
});
