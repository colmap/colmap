// Open external links in a new browser tab.
//
// Any anchor pointing to a different host than the current site (e.g. the
// GitHub repository, contributor profiles, or release downloads) gets
// target="_blank" plus rel="noopener noreferrer" so it opens in a new tab
// without exposing the opener. Same-site links (including absolute
// https://colmap.github.io/ URLs) are left untouched. Runs on every page and
// covers both reStructuredText links and raw-HTML anchors (e.g. the landing
// page hero buttons).
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    var anchors = document.querySelectorAll('a[href^="http://"], a[href^="https://"]');
    anchors.forEach(function (anchor) {
      if (anchor.hostname && anchor.hostname !== window.location.hostname) {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
      }
    });
  });
})();
