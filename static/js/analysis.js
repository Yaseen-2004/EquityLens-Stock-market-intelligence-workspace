/* Charts for the /analysis page. Loaded after dashboard.js, which already
   defined the theme handling; this file re-reads the same CSS tokens so both
   pages stay in one palette. */

(function () {
  "use strict";

  var el = document.getElementById("analysis-data");
  if (!el || typeof Chart === "undefined") return;

  var A;
  try {
    A = JSON.parse(el.textContent);
  } catch (e) {
    console.error("analysis: could not parse page data —", e.message);
    return;
  }

  var root = document.documentElement;
  function tok(n) { return getComputedStyle(root).getPropertyValue(n).trim(); }

  function tokens() {
    return {
      text: tok("--text-primary"),
      secondary: tok("--text-secondary"),
      muted: tok("--text-muted"),
      grid: tok("--grid"),
      surface: tok("--surface-1"),
      border: tok("--border"),
      pos: tok("--pos"),
      neg: tok("--neg"),
      s1: tok("--series-1"),
      s2: tok("--series-2"),
      seq: tok("--seq-450")
    };
  }

  function tip(t) {
    return {
      backgroundColor: t.surface, titleColor: t.text, bodyColor: t.secondary,
      borderColor: t.border, borderWidth: 1, padding: 11, cornerRadius: 8,
      displayColors: false, titleFont: { size: 13, weight: "600" },
      bodyFont: { size: 12.5 }
    };
  }

  var drawers = [];
  function redraw() { drawers.forEach(function (f) { f(); }); }

  // ------------------------------------------------------------ scatter ---
  var scatterCanvas = document.getElementById("scatter-chart");
  if (scatterCanvas) {
    var sc = null;
    var draw = function () {
      var t = tokens();
      if (sc) sc.destroy();

      var pts = A.rows.map(function (r) {
        return { x: r.vol, y: r.cagr, sym: r.symbol, name: r.name,
                 sharpe: r.sharpe, dd: r.max_dd };
      });
      var bench = A.benchmark_stats;
      var bPt = [{ x: bench.volatility_pct, y: bench.cagr_pct,
                   sym: A.benchmark, name: bench.name,
                   sharpe: bench.sharpe, dd: bench.max_drawdown_pct }];

      // Reference line through the origin and the benchmark: equal
      // return-per-unit-of-risk. Points above it did better per unit of risk.
      var maxX = Math.max.apply(null, pts.map(function (p) { return p.x; })) * 1.12;
      var slope = bench.cagr_pct / bench.volatility_pct;

      sc = new Chart(scatterCanvas.getContext("2d"), {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Reference",
              type: "line",
              data: [{ x: 0, y: 0 }, { x: maxX, y: slope * maxX }],
              borderColor: t.muted,
              borderWidth: 1.5,
              borderDash: [5, 5],
              pointRadius: 0,
              fill: false,
              order: 3
            },
            {
              label: "Companies",
              data: pts,
              backgroundColor: t.seq,
              borderColor: t.surface,
              borderWidth: 2,          // 2px surface ring on overlapping marks
              pointRadius: 9,          // >= 8px markers
              pointHoverRadius: 11,
              order: 1
            },
            {
              label: A.benchmark,
              data: bPt,
              backgroundColor: t.s2,
              borderColor: t.surface,
              borderWidth: 2,
              pointRadius: 10,
              pointStyle: "rectRot",   // shape, not just colour
              pointHoverRadius: 12,
              order: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { right: 26, top: 18, left: 4, bottom: 4 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              callbacks: {
                title: function (i) { return i[0].raw.sym + " - " + i[0].raw.name; },
                label: function (i) {
                  var r = i.raw;
                  return ["Return:   " + r.y.toFixed(1) + "% a year",
                          "Risk:     " + r.x.toFixed(1) + "% volatility",
                          "Sharpe:   " + r.sharpe.toFixed(2),
                          "Max fall: " + r.dd.toFixed(1) + "%"];
                }
              },
              filter: function (i) { return i.datasetIndex !== 0; }
            })
          },
          scales: {
            x: {
              title: { display: true, text: "Annualised volatility (%)",
                       color: t.muted, font: { size: 12 } },
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 },
                       callback: function (v) { return v + "%"; } }
            },
            y: {
              title: { display: true, text: "Annualised return (%)",
                       color: t.muted, font: { size: 12 } },
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 },
                       callback: function (v) { return v + "%"; } }
            }
          },
          animation: { duration: 420 }
        },
        plugins: [{
          // Every point is direct-labelled: identity is never colour-alone.
          id: "pointLabels",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx;
            ctx.save();
            ctx.font = "600 11.5px ui-sans-serif, system-ui, 'Segoe UI', sans-serif";
            ctx.fillStyle = tokens().text;
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";
            [1, 2].forEach(function (di) {
              var meta = c.getDatasetMeta(di);
              if (!meta || meta.hidden) return;
              meta.data.forEach(function (pt, i) {
                var d = c.data.datasets[di].data[i];
                ctx.fillText(d.sym, pt.x, pt.y - 13);
              });
            });
            ctx.restore();
          }
        }]
      });
    };
    drawers.push(draw);
    draw();
  }

  // ------------------------------------------------- magnitude bar charts --
  function barChart(canvasId, pick, fmt, colorFn, ascending) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ch = null;

    var draw = function () {
      var t = tokens();
      if (ch) ch.destroy();

      // Sorted so the bar the heading is about comes first.
      var rows = A.rows.slice().sort(function (a, b) {
        return ascending ? pick(a) - pick(b) : pick(b) - pick(a);
      });
      var values = rows.map(pick);

      ch = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: rows.map(function (r) { return r.symbol; }),
          datasets: [{
            data: values,
            // Nominal categories -> one colour. No value ramp.
            backgroundColor: colorFn(t),
            borderRadius: 4,
            borderSkipped: false,
            borderColor: t.surface,
            borderWidth: { left: 1, right: 1 },
            barPercentage: 0.82,
            categoryPercentage: 0.86
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { right: 62, left: 2 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              callbacks: {
                title: function (i) {
                  var r = rows[i[0].dataIndex];
                  return r.symbol + " - " + r.name;
                },
                label: function (i) {
                  var r = rows[i.dataIndex];
                  return ["Total:    " + r.total.toFixed(1) + "%",
                          "CAGR:     " + r.cagr.toFixed(1) + "%",
                          "Vol:      " + r.vol.toFixed(1) + "%",
                          "Sharpe:   " + r.sharpe.toFixed(2),
                          "Max fall: " + r.max_dd.toFixed(1) + "%"];
                }
              }
            })
          },
          scales: {
            x: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 },
                       callback: function (v) { return fmt(v); } }
            },
            y: {
              grid: { display: false },
              border: { display: false },
              ticks: { color: t.secondary, font: { size: 12.5, weight: "600" } }
            }
          },
          animation: { duration: 420 }
        },
        plugins: [{
          id: canvasId + "-labels",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx, meta = c.getDatasetMeta(0);
            ctx.save();
            ctx.font = "600 12px ui-sans-serif, system-ui, 'Segoe UI', sans-serif";
            ctx.fillStyle = tokens().text;
            ctx.textBaseline = "middle";
            meta.data.forEach(function (bar, i) {
              var neg = values[i] < 0;
              ctx.textAlign = neg ? "right" : "left";
              ctx.fillText(fmt(values[i]), bar.x + (neg ? -8 : 8), bar.y);
            });
            ctx.restore();
          }
        }]
      });
    };
    drawers.push(draw);
    draw();
  }

  var pct = function (v) { return v.toFixed(1) + "%"; };

  barChart("cagr-chart", function (r) { return r.cagr; }, pct,
           function (t) { return t.seq; });
  // Drawdowns are all losses, so the loss colour carries meaning here, and
  // the deepest fall leads - the heading is about the worst.
  barChart("dd-chart", function (r) { return r.max_dd; }, pct,
           function (t) { return t.neg; }, true);

  // ------------------------------------------------------------- growth ---
  var growthCanvas = document.getElementById("growth-chart");
  var growthLegend = document.getElementById("growth-legend");
  if (growthCanvas && A.growth) {
    var gc = null;
    var names = Object.keys(A.growth.series);

    var draw = function () {
      var t = tokens();
      if (gc) gc.destroy();
      var palette = [t.s1, t.s2];

      gc = new Chart(growthCanvas.getContext("2d"), {
        type: "line",
        data: {
          labels: A.growth.dates,
          datasets: names.map(function (n, i) {
            return {
              label: n,
              data: A.growth.series[n],
              borderColor: palette[i % palette.length],
              backgroundColor: palette[i % palette.length],
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              pointHoverBorderWidth: 2,
              pointHoverBorderColor: t.surface,
              tension: 0.15
            };
          })
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          layout: { padding: { right: 12, top: 4 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              displayColors: true,
              callbacks: {
                title: function (i) {
                  return new Date(i[0].label).toLocaleDateString(undefined,
                    { year: "numeric", month: "short", day: "numeric" });
                },
                label: function (i) {
                  return i.dataset.label + "  $" + i.parsed.y.toFixed(2);
                }
              }
            })
          },
          scales: {
            x: {
              grid: { display: false },
              border: { color: t.grid },
              ticks: {
                color: t.muted, font: { size: 11.5 }, maxRotation: 0,
                autoSkip: true, maxTicksLimit: 8,
                callback: function (v) {
                  var d = new Date(this.getLabelForValue(v));
                  return d.toLocaleDateString(undefined,
                    { month: "short", year: "2-digit" });
                }
              }
            },
            y: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 }, padding: 8,
                       callback: function (v) { return "$" + v; } }
            }
          },
          animation: { duration: 420 }
        }
      });

      if (growthLegend) {
        growthLegend.innerHTML = names.map(function (n, i) {
          return '<span class="legend-item"><span class="legend-key" style="background:' +
                 palette[i % palette.length] + '"></span>' + n + "</span>";
        }).join("");
      }
    };
    drawers.push(draw);
    draw();
  }

  // Keep the analysis charts in step with the theme toggle.
  var toggle = document.getElementById("theme-toggle");
  if (toggle) toggle.addEventListener("click", function () { setTimeout(redraw, 0); });
  window.matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", function () {
          if (!root.getAttribute("data-theme")) redraw();
        });
})();
