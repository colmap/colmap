import {defineConfig} from "vitest/config";

export default defineConfig({
  base: "./",
  build: {
    outDir: "_static/viewer",
    emptyOutDir: true,
    lib: {
      entry: "viewer_src/main.ts",
      formats: ["es"],
      fileName: () => "viewer.js",
    },
    rollupOptions: {
      output: {
        assetFileNames: "[name][extname]",
        chunkFileNames: "[name]-[hash].js",
      },
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
