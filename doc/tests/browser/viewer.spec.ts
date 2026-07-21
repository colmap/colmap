import {expect, test} from "@playwright/test";

import {cameraFile, imageFile, pointFile} from "../binary_fixture";

test("initializes the local-only viewer shell", async ({page}) => {
  await page.goto("/tests/viewer.html");
  await expect(page.getByRole("heading", {name: "Open a COLMAP reconstruction"})).toBeVisible();
  await expect(page.getByRole("button", {name: "Choose folder"})).toBeVisible();
  await expect(page.locator("#viewer-canvas")).toBeAttached();
  await expect(page.locator("#viewer-stats")).toHaveText("No model loaded");

  const files = await Promise.all([cameraFile(), imageFile(), pointFile()].map(async (file) => ({
    name: file.name,
    mimeType: "application/octet-stream",
    buffer: Buffer.from(await file.arrayBuffer()),
  })));
  await page.locator("#viewer-folder-input").setInputFiles(files);
  await expect(page.locator("#viewer-stats")).toHaveText("1 cameras / 1 points");
  await expect(page.getByRole("heading", {name: "Model loaded"})).toBeVisible();
  await page.locator("#viewer-projection").selectOption("orthographic");
  await expect(page.locator("#viewer-projection")).toHaveValue("orthographic");

  const malformedFiles = files.map((file) => file.name === "cameras.bin" ? {...file, buffer: Buffer.from([1])} : file);
  await page.locator("#viewer-folder-input").setInputFiles(malformedFiles);
  await expect(page.locator("#viewer-stats")).toHaveText("No model loaded");
  await expect(page.getByRole("heading", {name: "Open a COLMAP reconstruction"})).toBeVisible();
  await expect(page.locator("#viewer-status")).toContainText("Failed to parse model");
  await expect(page.locator("#viewer-reset")).toBeDisabled();
});
