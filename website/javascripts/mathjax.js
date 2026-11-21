// MathJax 3 configuration for pymdownx.arithmatex (generic mode), as recommended by Material for MkDocs.
// The docstrings write bold vectors as \bm{...}; MathJax has no `bm` package, so it is mapped to \boldsymbol.
// The boldsymbol extension is loaded and enabled explicitly: left to autoload, it failed on the built site
// and \boldsymbol showed as undefined (red).
window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      bm: ["\\boldsymbol{#1}", 1],
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
