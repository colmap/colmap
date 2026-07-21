// Interactive install selector for the COLMAP landing page.
//
// Renders a pytorch.org-style grid: the user picks an operating system, an
// install method, and a compute backend (CUDA vs CPU), and the widget shows the
// exact command or download link. The script is loaded on every page but only
// activates when the landing page's mount point (#colmap-install-selector) is
// present.
(function () {
  "use strict";

  // Helpers to build the per-cell result shown in the output box.
  function cmd(text, note) {
    return { kind: "command", text: text, note: note || null };
  }
  function link(text, url, note) {
    return { kind: "link", text: text, url: url, note: note || null };
  }

  // Install matrix: MATRIX[os].methods keeps the display order; MATRIX[os].cells
  // maps method -> { cuda, cpu }, where each entry is a result object or null
  // (unavailable / disabled).
  var MATRIX = {
    linux: {
      label: "Linux",
      methods: ["pip", "conda", "docker", "binary", "source"],
      cells: {
        pip: {
          cuda: cmd("pip install pycolmap-cuda12"),
          cpu: cmd("pip install pycolmap"),
        },
        conda: {
          cuda: null,
          cpu: cmd("conda install -c conda-forge colmap"),
        },
        docker: {
          cuda: cmd(
            "docker pull colmap/colmap:latest",
            "Run with GPU flags (NVIDIA Container Toolkit) to enable CUDA."
          ),
          cpu: cmd(
            "docker pull colmap/colmap:latest",
            "The image is CUDA-based but also runs CPU-only."
          ),
        },
        binary: {
          cuda: null,
          cpu: link(
            "Distribution packages (Repology)",
            "https://repology.org/metapackage/colmap/versions",
            "Distro packages ship without CUDA. For GPU support, build from source."
          ),
        },
        source: {
          cuda: link(
            "Build from source (with CUDA)",
            "https://colmap.github.io/install.html#debian-ubuntu",
            "Install the CUDA toolkit, then configure COLMAP with CUDA enabled."
          ),
          cpu: link(
            "Build from source",
            "https://colmap.github.io/install.html#debian-ubuntu"
          ),
        },
      },
    },
    macos: {
      label: "macOS",
      methods: ["pip", "conda", "brew", "docker", "source"],
      cells: {
        pip: { cuda: null, cpu: cmd("pip install pycolmap") },
        conda: { cuda: null, cpu: cmd("conda install -c conda-forge colmap") },
        brew: { cuda: null, cpu: cmd("brew install colmap") },
        docker: {
          cuda: null,
          cpu: cmd(
            "docker pull colmap/colmap:latest",
            "GPU acceleration is not available on macOS."
          ),
        },
        source: {
          cuda: null,
          cpu: link(
            "Build from source",
            "https://colmap.github.io/install.html#mac"
          ),
        },
      },
    },
    windows: {
      label: "Windows",
      methods: ["binary", "pip", "conda", "vcpkg", "docker"],
      cells: {
        binary: {
          cuda: link(
            "Download from GitHub Releases",
            "https://github.com/colmap/colmap/releases",
            "Use the colmap-x64-windows-cuda package."
          ),
          cpu: link(
            "Download from GitHub Releases",
            "https://github.com/colmap/colmap/releases",
            "Use the colmap-x64-windows-nocuda package."
          ),
        },
        pip: {
          cuda: null,
          cpu: cmd(
            "pip install pycolmap",
            "Windows wheels are CPU-only. For GPU, use the binary package or vcpkg."
          ),
        },
        conda: { cuda: null, cpu: cmd("conda install -c conda-forge colmap") },
        vcpkg: {
          cuda: cmd("vcpkg install colmap[cuda,tests]:x64-windows"),
          cpu: cmd("vcpkg install colmap:x64-windows"),
        },
        docker: {
          cuda: cmd(
            "docker pull colmap/colmap:latest",
            "Requires WSL2 with the NVIDIA Container Toolkit for GPU support."
          ),
          cpu: cmd("docker pull colmap/colmap:latest"),
        },
      },
    },
  };

  var OS_ORDER = ["linux", "macos", "windows"];
  var METHOD_LABELS = {
    pip: "pip",
    conda: "Conda",
    docker: "Docker",
    binary: "Binary",
    brew: "Homebrew",
    vcpkg: "vcpkg",
    source: "Source",
  };
  var COMPUTE_ORDER = ["cuda", "cpu"];
  var COMPUTE_LABELS = { cuda: "CUDA (GPU)", cpu: "CPU" };

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Current selection.
  var state = { os: "linux", method: "pip", compute: "cuda" };

  function computesFor(os, method) {
    var cell = MATRIX[os].cells[method];
    return COMPUTE_ORDER.filter(function (c) {
      return cell && cell[c];
    });
  }

  // Coerce the selection into a valid (os, method, compute) triple.
  function normalize() {
    if (!MATRIX[state.os]) state.os = "linux";
    var methods = MATRIX[state.os].methods;
    if (methods.indexOf(state.method) === -1) state.method = methods[0];
    var avail = computesFor(state.os, state.method);
    if (avail.indexOf(state.compute) === -1) state.compute = avail[0];
  }

  function button(label, active, disabled, onClick) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "cis__btn" +
      (active ? " cis__btn--active" : "") +
      (disabled ? " cis__btn--disabled" : "");
    btn.textContent = label;
    if (disabled) {
      btn.setAttribute("aria-disabled", "true");
      btn.disabled = true;
    } else {
      btn.addEventListener("click", onClick);
    }
    btn.setAttribute("aria-pressed", active ? "true" : "false");
    return btn;
  }

  function row(root, labelText, options) {
    var r = document.createElement("div");
    r.className = "cis__row";
    var lbl = document.createElement("div");
    lbl.className = "cis__label";
    lbl.textContent = labelText;
    var opts = document.createElement("div");
    opts.className = "cis__options";
    options.forEach(function (o) {
      opts.appendChild(o);
    });
    r.appendChild(lbl);
    r.appendChild(opts);
    root.appendChild(r);
  }

  function render(root) {
    normalize();
    root.innerHTML = "";

    // OS row.
    row(
      root,
      "OS",
      OS_ORDER.map(function (os) {
        return button(MATRIX[os].label, state.os === os, false, function () {
          state.os = os;
          render(root);
        });
      })
    );

    // Method row.
    row(
      root,
      "Package",
      MATRIX[state.os].methods.map(function (m) {
        return button(METHOD_LABELS[m], state.method === m, false, function () {
          state.method = m;
          render(root);
        });
      })
    );

    // Compute row (disable unavailable backends for the current os+method).
    var avail = computesFor(state.os, state.method);
    row(
      root,
      "Compute",
      COMPUTE_ORDER.map(function (c) {
        var disabled = avail.indexOf(c) === -1;
        return button(
          COMPUTE_LABELS[c],
          state.compute === c,
          disabled,
          function () {
            state.compute = c;
            render(root);
          }
        );
      })
    );

    // Output box.
    var result = MATRIX[state.os].cells[state.method][state.compute];
    var out = document.createElement("div");
    out.className = "cis__output";

    if (result && result.kind === "command") {
      var box = document.createElement("div");
      box.className = "cis__cmd";
      var code = document.createElement("code");
      code.innerHTML =
        '<span class="cis__prompt">$</span> ' + escapeHtml(result.text);
      var copy = document.createElement("button");
      copy.type = "button";
      copy.className = "cis__copy";
      copy.textContent = "Copy";
      copy.addEventListener("click", function () {
        var done = function () {
          copy.textContent = "Copied!";
          setTimeout(function () {
            copy.textContent = "Copy";
          }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(result.text).then(done, function () {});
        } else {
          done();
        }
      });
      box.appendChild(code);
      box.appendChild(copy);
      out.appendChild(box);
    } else if (result && result.kind === "link") {
      var a = document.createElement("a");
      a.className = "cis__link";
      a.href = result.url;
      a.textContent = result.text;
      a.rel = "noopener";
      out.appendChild(a);
    }

    if (result && result.note) {
      var note = document.createElement("p");
      note.className = "cis__note";
      note.textContent = result.note;
      out.appendChild(note);
    }

    root.appendChild(out);
  }

  function init() {
    var root = document.getElementById("colmap-install-selector");
    if (!root) return;
    root.classList.add("cis");
    render(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
