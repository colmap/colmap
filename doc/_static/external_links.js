// Open external links in a new browser tab.
//
// Any anchor pointing to a different host than the current site (e.g. the
// GitHub repository, contributor profiles, or release downloads) gets
// target="_blank" plus rel="noopener noreferrer" so it opens in a new tab
// without exposing the opener. Same-site links (including absolute
// https://colmap.github.io/ URLs) are left untouched. Covers both
// reStructuredText links and raw-HTML anchors (e.g. the landing page hero
// buttons), as well as anchors injected after load (e.g. the install
// selector's download links), which a MutationObserver picks up.
(function () {
  "use strict";

  function markExternal(anchor) {
    if (anchor.hostname && anchor.hostname !== window.location.hostname) {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    }
  }

  function scan(root) {
    var anchors = root.querySelectorAll(
      'a[href^="http://"], a[href^="https://"]'
    );
    anchors.forEach(markExternal);
  }

  document.addEventListener("DOMContentLoaded", function () {
    scan(document);

    // Re-scan anchors added after the initial load (e.g. the install
    // selector rebuilds its output on every click).
    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
          if (node.nodeType !== Node.ELEMENT_NODE) return;
          if (node.tagName === "A") markExternal(node);
          if (node.querySelectorAll) scan(node);
        });
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
