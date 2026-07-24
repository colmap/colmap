import {defineConfig} from "vitest/config";

export default defineConfig({
  base: "./",
  build: {
    outDir: "_static/viewer",
    emptyOutDir: true,
    lib: {
      entry: {
        component: "viewer_src/main.ts",
        viewer: "viewer_src/auto_mount.ts",
      },
      cssFileName: "viewer",
      formats: ["es"],
      fileName: (_format, entryName) => `${entryName}.js`,
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
