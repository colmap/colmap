import {defineConfig} from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  use: {baseURL: "http://127.0.0.1:4173", channel: "chromium"},
  webServer: {
    command: "vite --host 127.0.0.1 --port 4173",
    port: 4173,
    reuseExistingServer: true,
  },
});
