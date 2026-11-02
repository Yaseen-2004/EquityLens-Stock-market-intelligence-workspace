/* Charts for the /forecast page, plus the background re-run control. */

(function () {
  "use strict";

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

  // ------------------------------------------------------- the re-run job --
  var runBtn = document.getElementById("fc-run");
  var statusEl = document.getElementById("fc-status");

  function showStatus(msg) {
    if (!statusEl) return;
    statusEl.hidden = false;
    statusEl.textContent = msg;
  }

  if (runBtn) {
    runBtn.addEventListener("click", function () {
      runBtn.disabled = true;
      showStatus("Starting the backtest…");
      fetch("/api/forecast/run", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (j) {
          showStatus(j.message);
          var poll = setInterval(function () {
            fetch("/api/forecast/status")
              .then(function (r) { return r.json(); })
              .then(function (s) {
                showStatus(s.message);
                if (!s.running) {
                  clearInterval(poll);
                  runBtn.disabled = false;
                  showStatus(s.message + " — reloading…");
                  setTimeout(function () { location.reload(); }, 1200);
                }
              });
          }, 3000);
        })
        .catch(function () {
          showStatus("Could not start the backtest.");
          runBtn.disabled = false;
        });
    });
  }

  // --------------------------------------------------------------- charts --
  var el = document.getElementById("forecast-data");
  if (!el || typeof Chart === "undefined") return;

  var F;
  try {
    F = JSON.parse(el.textContent);
  } catch (e) {
    // A silent return here once hid a NaN in the payload for an entire
    // debugging session - every chart blank, no error anywhere.
    console.error("forecast: could not parse page data —", e.message);
    return;
  }

  var agg = F.aggregate || {};
  var names = Object.keys(agg);

  /* Skill: a diverging measure around zero, so the sign carries the meaning -
     blue beat the benchmark, red did worse than doing nothing. */
  var skillCanvas = document.getElementById("skill-chart");
  if (skillCanvas) {
    var sc = null;
    var draw = function () {
      var t = tokens();
      if (sc) sc.destroy();
      var rows = names.filter(function (n) { return n !== "naive"; });
      var values = rows.map(function (n) { return agg[n].skill_vs_naive; });

      sc = new Chart(skillCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: rows,
          datasets: [{
            data: values,
            backgroundColor: values.map(function (v) {
              return v > 0 ? t.pos : t.neg;
            }),
            borderRadius: 4,
            borderSkipped: false,
            borderColor: t.surface,
            borderWidth: { left: 1, right: 1 },
            barPercentage: 0.72,
            categoryPercentage: 0.8
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { right: 58, left: 2 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              callbacks: {
                label: function (i) {
                  var s = agg[rows[i.dataIndex]];
                  return ["Skill: " + s.skill_vs_naive.toFixed(3),
                          "MAPE:  " + s.mape.toFixed(2) + "%",
                          "Beat naive on " + s.beat_naive_count + "/" + s.n];
                }
              }
            })
          },
          scales: {
            x: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 } }
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
          id: "skillLabels",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx, meta = c.getDatasetMeta(0), tk = tokens();
            ctx.save();
            ctx.font = "600 12px ui-sans-serif, system-ui, 'Segoe UI', sans-serif";
            ctx.fillStyle = tk.text;
            ctx.textBaseline = "middle";
            meta.data.forEach(function (bar, i) {
              var neg = values[i] < 0;
              var label = values[i].toFixed(3);
              var x = bar.x + (neg ? -8 : 8);
              // A long bar can push its label off the plot and onto the axis
              // labels; flip it inside the bar instead.
              if (neg && x - ctx.measureText(label).width < c.chartArea.left) {
                ctx.textAlign = "left";
                ctx.fillStyle = tk.surface;
                x = bar.x + 8;
              } else {
                ctx.textAlign = neg ? "right" : "left";
                ctx.fillStyle = tk.text;
              }
              ctx.fillText(label, x, bar.y);
            });
            ctx.restore();
          }
        }]
      });
    };
    drawers.push(draw);
    draw();
  }

  /* Direction: one hue, with a 50% coin-flip reference line - the whole point
     is how close to chance these are. */
  var dirCanvas = document.getElementById("dir-chart");
  if (dirCanvas) {
    var dc = null;
    var draw2 = function () {
      var t = tokens();
      if (dc) dc.destroy();
      var rows = names.filter(function (n) {
        return n !== "naive" && typeof agg[n].direction_pct === "number";
      });
      var actual = rows.map(function (n) { return agg[n].direction_pct; });
      // Plot the gap from a coin flip, not the raw percentage: a bar chart
      // whose axis starts at 45% would make a 2-point gap look decisive.
      var values = actual.map(function (v) { return v - 50; });

      dc = new Chart(dirCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: rows,
          datasets: [{
            data: values,
            backgroundColor: values.map(function (v) {
              return v > 0 ? t.pos : t.neg;
            }),
            borderRadius: 4,
            borderSkipped: false,
            borderColor: t.surface,
            borderWidth: { left: 1, right: 1 },
            barPercentage: 0.72,
            categoryPercentage: 0.8
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { top: 18 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              callbacks: {
                label: function (i) {
                  var a = actual[i.dataIndex];
                  return [a.toFixed(1) + "% correct direction",
                          (a >= 50 ? "+" : "") + (a - 50).toFixed(1)
                            + " points vs a coin flip"];
                }
              }
            })
          },
          scales: {
            x: {
              grid: { display: false },
              border: { color: t.grid },
              ticks: { color: t.secondary, font: { size: 12.5, weight: "600" } }
            },
            y: {
              suggestedMin: -6,
              suggestedMax: 6,
              title: { display: true,
                       text: "Percentage points vs a coin flip",
                       color: t.muted, font: { size: 12 } },
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 },
                       callback: function (v) {
                         return (v > 0 ? "+" : "") + v;
                       } }
            }
          },
          animation: { duration: 420 }
        },
        plugins: [{
          id: "coinFlip",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx, y = c.scales.y.getPixelForValue(0), tk = tokens();
            ctx.save();
            ctx.strokeStyle = tk.muted;
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(c.chartArea.left, y);
            ctx.lineTo(c.chartArea.right, y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
            ctx.fillStyle = tk.muted;
            // Anchored right: the leftmost bar sits on top of this otherwise.
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            ctx.fillText("50% — a coin flip", c.chartArea.right - 4, y - 5);

            // Direct-label each bar with the real accuracy.
            ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
            ctx.fillStyle = tk.text;
            ctx.textAlign = "center";
            c.getDatasetMeta(0).data.forEach(function (bar, i) {
              var up = values[i] >= 0;
              ctx.textBaseline = up ? "bottom" : "top";
              ctx.fillText(actual[i].toFixed(1) + "%", bar.x,
                           bar.y + (up ? -6 : 6));
            });
            ctx.restore();
          }
        }]
      });
    };
    drawers.push(draw2);
    draw2();
  }

  /* Projected ranges: a floating bar per company showing the 80% band as a
     percentage of the last close, with the point estimate marked. Percentages
     keep companies at very different prices on one honest axis. */
  var bandCanvas = document.getElementById("band-chart");
  if (bandCanvas) {
    var bc = null;
    var draw3 = function () {
      var t = tokens();
      if (bc) bc.destroy();

      var syms = [], ranges = [], points = [], meta = [];
      Object.keys(F.results).forEach(function (sym) {
        var p = F.results[sym].projection;
        if (!p || p.error) return;
        syms.push(sym);
        ranges.push([(p.lo80 / p.last_close - 1) * 100,
                     (p.hi80 / p.last_close - 1) * 100]);
        points.push(p.change_pct);
        meta.push(p);
      });

      bc = new Chart(bandCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: syms,
          datasets: [{
            label: "80% range",
            data: ranges,
            backgroundColor: t.seq + "55",
            borderColor: t.seq,
            borderWidth: 1,
            borderRadius: 4,
            barPercentage: 0.6,
            categoryPercentage: 0.8
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { right: 16, left: 2 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tip(t), {
              callbacks: {
                title: function (i) { return syms[i[0].dataIndex]; },
                label: function (i) {
                  var p = meta[i.dataIndex];
                  return ["Last close: $" + p.last_close.toFixed(2),
                          "Point:      $" + p.point.toFixed(2) +
                            "  (" + p.change_pct.toFixed(2) + "%)",
                          "80% range:  $" + p.lo80.toFixed(0) + " – $" + p.hi80.toFixed(0),
                          "95% range:  $" + p.lo95.toFixed(0) + " – $" + p.hi95.toFixed(0)];
                }
              }
            })
          },
          scales: {
            x: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              title: { display: true, text: "Change from last close (%)",
                       color: t.muted, font: { size: 12 } },
              ticks: { color: t.muted, font: { size: 11.5 },
                       callback: function (v) { return v + "%"; } }
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
          // Zero line, and a tick for each point estimate inside its band.
          id: "bandMarks",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx, tk = tokens();
            var zero = c.scales.x.getPixelForValue(0);
            ctx.save();
            ctx.strokeStyle = tk.muted;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(zero, c.chartArea.top);
            ctx.lineTo(zero, c.chartArea.bottom);
            ctx.stroke();

            var meta0 = c.getDatasetMeta(0);
            meta0.data.forEach(function (bar, i) {
              var x = c.scales.x.getPixelForValue(points[i]);
              ctx.strokeStyle = points[i] >= 0 ? tk.pos : tk.neg;
              ctx.lineWidth = 2.5;
              ctx.beginPath();
              ctx.moveTo(x, bar.y - 9);
              ctx.lineTo(x, bar.y + 9);
              ctx.stroke();
            });
            ctx.restore();
          }
        }]
      });
    };
    drawers.push(draw3);
    draw3();
  }

  var toggle = document.getElementById("theme-toggle");
  if (toggle) toggle.addEventListener("click", function () { setTimeout(redraw, 0); });
  window.matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", function () {
          if (!root.getAttribute("data-theme")) redraw();
        });
})();
