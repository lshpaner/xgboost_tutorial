# How XGBoost Works

An interactive guide to regularized gradient boosting, built as a static site.

Live: https://lshpaner.github.io/xgboost_tutorial/

## What it covers

The page works through the regularized objective, the closed-form leaf weight
`w* = -G / (H + lambda)`, and the split gain that follows from it. It then covers the
contributions specific to Chen and Guestrin (2016) that generic boosting write-ups tend to
omit: sparsity-aware split finding, the hessian-weighted quantile sketch, the column block
layout, and the two forms of subsampling. Closing sections give a tuning priority order, an
honest account of where boosted trees lose, and four evaluation traps that show up
repeatedly in applied work.

## The demo

`script.js` contains a teaching implementation of gradient boosting in about 120 lines:

- one feature, squared error loss, so `g = F - y` and `h = 1`
- exact greedy split search over every candidate threshold
- split selection by `Gain = 0.5 * [GL^2/(HL+L) + GR^2/(HR+L) - G^2/(H+L)] - gamma`
- leaf weights by `-G / (H + lambda)`

The lambda and gamma sliders therefore act on the model the way they do in the library.
Raising lambda shrinks leaf weights toward zero; raising gamma removes low-gain splits and
the leaf count in the metrics row drops. There is no quantile sketch, no sparsity handling,
no subsampling, and none of the systems work from the paper.

## Files

```
xgboost_tutorial/
├── index.html    # content and structure
├── styles.css    # all styling, no build step
├── script.js     # boosting implementation, plots, tree drawing, navigation
└── README.md
```

## Dependencies

Three external scripts, all pinned to exact versions:

| Library | Version | Used for |
| --- | --- | --- |
| Plotly.js | 2.35.2 | fit, residual, and learning-curve plots |
| D3 | 7.9.0 | tree diagrams |
| MathJax | 3.2.2 | LaTeX rendering |

Pinning is deliberate. An earlier version of this page loaded
`https://polyfill.io/v3/polyfill.min.js?features=es6`, copied from the MathJax setup
snippet. That domain was sold in 2024, began serving malicious code, and in mid-2026
started returning HTTP 401 responses, which made browsers show a native credential prompt
on every visit. The reference has been removed, along with an unused Math.js dependency and
the AOS animation library.

**Remaining hardening step:** vendor the three files into the repository, or add
Subresource Integrity hashes, so a future CDN compromise cannot reach visitors.

## Running locally

No build step. Any static server works:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Deployment

Push to a repository with GitHub Pages enabled on the branch root. The CDN cache clears
within a few minutes of a push, so hard-refresh when verifying a change.

## References

1. Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD '16*, 785-794.
2. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29(5), 1189-1232.
3. Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics and Data Analysis* 38(4), 367-378.

## Author

Leon Shpaner

## License

Free to use and adapt for teaching, with attribution. Add a `LICENSE` file if you want
something enforceable; CC BY 4.0 fits the prose and MIT fits the code.
