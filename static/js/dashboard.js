/* Chart rendering. Colours are read from the CSS custom properties defined in
   style.css, so light/dark is decided in exactly one place and the charts
   redraw themselves when the theme changes. */

(function () {
  "use strict";

  // ---------------------------------------------------------------- theme --
  var root = document.documentElement;

  function readToken(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function tokens() {
    return {
      text: readToken("--text-primary"),
      secondary: readToken("--text-secondary"),
      muted: readToken("--text-muted"),
      grid: readToken("--grid"),
      surface: readToken("--surface-1"),
      border: readToken("--border"),
      pos: readToken("--pos"),
      neg: readToken("--neg"),
      seq: [readToken("--seq-500"), readToken("--seq-450"), readToken("--seq-400"),
            readToken("--seq-300"), readToken("--seq-200")],
      series: [readToken("--series-1"), readToken("--series-2"),
               readToken("--series-3"), readToken("--series-4"),
               readToken("--series-5")]
    };
  }

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) { /* private mode */ }
    if (saved === "dark" || saved === "light") root.setAttribute("data-theme", saved);
    else root.removeAttribute("data-theme");
  }

  function currentlyDark() {
    var explicit = root.getAttribute("data-theme");
    if (explicit === "dark") return true;
    if (explicit === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  initTheme();

  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentlyDark() ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
      redrawAll();
    });
  }
  window.matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", function () {
          if (!root.getAttribute("data-theme")) redrawAll();
        });

  // -------------------------------------------------------------- helpers --
  function readJSON(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      console.error("dashboard: could not parse " + id + " —", e.message);
      return null;
    }
  }

  function money(n) {
    return "$" + n.toLocaleString(undefined,
      { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function compact(n) {
    var abs = Math.abs(n);
    if (abs >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
    if (abs >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
    if (abs >= 1e6) return "$" + (n / 1e6).toFixed(1) + "M";
    return money(n);
  }

  function volume(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
    return String(n);
  }

  // Shared tooltip styling so every chart on the site reads the same.
  function tooltipStyle(t) {
    return {
      backgroundColor: t.surface,
      titleColor: t.text,
      bodyColor: t.secondary,
      borderColor: t.border,
      borderWidth: 1,
      padding: 11,
      cornerRadius: 8,
      displayColors: false,
      titleFont: { size: 13, weight: "600" },
      bodyFont: { size: 12.5 }
    };
  }

  var charts = [];
  function register(fn) { charts.push(fn); }
  function redrawAll() { charts.forEach(function (f) { f(); }); }

  // ------------------------------------------------------- ranking chart ---
  var rankData = readJSON("rank-data");
  var rankCanvas = document.getElementById("rank-chart");

  if (rankData && rankCanvas && rankData.length) {
    var mode = window.RANK_MODE || "market_cap";
    var chart = null;

    var valueOf = {
      market_cap: function (c) { return c.market_cap || 0; },
      price: function (c) { return c.last_close; },
      ytd: function (c) { return c.ytd_pct; },
      volume: function (c) { return c.avg_volume; }
    }[mode];

    var formatOf = {
      market_cap: compact,
      price: money,
      ytd: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "%"; },
      volume: volume
    }[mode];

    var draw = function () {
      var t = tokens();
      if (chart) chart.destroy();

      var values = rankData.map(valueOf);
      var diverging = mode === "ytd";

      // Companies are NOMINAL categories, so a darker-where-bigger ramp would
      // double-encode bar length as lightness and burn the only free channel.
      // One series -> one colour. Polarity (YTD) is the exception: there the
      // sign is real information, so the diverging poles earn their place.
      var colors = rankData.map(function (c) {
        if (diverging) return valueOf(c) >= 0 ? t.pos : t.neg;
        return t.seq[1];   // sequential slot 450 - the single series colour
      });

      chart = new Chart(rankCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: rankData.map(function (c) { return c.symbol; }),
          datasets: [{
            label: "value",
            data: values,
            backgroundColor: colors,
            hoverBackgroundColor: colors,
            borderRadius: 4,          // 4px rounded data-end
            borderSkipped: false,
            // 2px surface gap between adjacent bars
            borderColor: t.surface,
            borderWidth: { top: 0, bottom: 0, left: 1, right: 1 },
            barPercentage: 0.82,
            categoryPercentage: 0.86
          }]
        },
        options: {
          indexAxis: "y",             // horizontal: long company names fit
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { right: 68, left: 2, top: 2, bottom: 2 } },
          plugins: {
            legend: { display: false },   // single series: title names it
            tooltip: Object.assign(tooltipStyle(t), {
              callbacks: {
                title: function (items) {
                  var c = rankData[items[0].dataIndex];
                  return "#" + c.rank + "  " + c.symbol + " - " + c.name;
                },
                label: function (item) {
                  var c = rankData[item.dataIndex];
                  return [
                    "Last close:  " + money(c.last_close),
                    "YTD:         " + (c.ytd_pct >= 0 ? "+" : "") + c.ytd_pct.toFixed(2) + "%",
                    "Market cap:  " + (c.market_cap ? compact(c.market_cap) : "n/a"),
                    "Avg volume:  " + volume(c.avg_volume)
                  ];
                }
              }
            }),
            // Direct value labels: required relief for the contrast WARN,
            // and they remove the need to read values off the axis.
            datalabels: false
          },
          scales: {
            x: {
              beginAtZero: true,
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: {
                color: t.muted, font: { size: 11.5 }, padding: 6,
                callback: function (v) { return formatOf(v); }
              }
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
          // Value printed at the end of each bar.
          id: "endLabels",
          afterDatasetsDraw: function (c) {
            var ctx = c.ctx;
            var meta = c.getDatasetMeta(0);
            ctx.save();
            ctx.font = "600 12px ui-sans-serif, system-ui, 'Segoe UI', sans-serif";
            ctx.fillStyle = tokens().text;
            ctx.textBaseline = "middle";
            meta.data.forEach(function (bar, i) {
              var v = values[i];
              var neg = v < 0;
              ctx.textAlign = neg ? "right" : "left";
              ctx.fillText(formatOf(v), bar.x + (neg ? -8 : 8), bar.y);
            });
            ctx.restore();
          }
        }]
      });
    };

    register(draw);
    draw();
  }

  // --------------------------------------------------------- trend chart ---
  var trendData = readJSON("trend-data");
  var trendCanvas = document.getElementById("trend-chart");
  var trendLegend = document.getElementById("trend-legend");

  if (trendData && trendCanvas && Object.keys(trendData).length) {
    var tChart = null;
    var symbols = Object.keys(trendData);

    var drawTrend = function () {
      var t = tokens();
      if (tChart) tChart.destroy();

      // Rebase every series to 100 so one axis serves them all.
      var datasets = symbols.map(function (sym, i) {
        var closes = trendData[sym].closes;
        var base = closes[0] || 1;
        return {
          label: sym,
          data: closes.map(function (c) { return (c / base) * 100; }),
          borderColor: t.series[i % t.series.length],
          backgroundColor: t.series[i % t.series.length],
          borderWidth: 2,             // 2px lines
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: t.surface,
          tension: 0.18
        };
      });

      var labels = trendData[symbols[0]].dates;

      tChart = new Chart(trendCanvas.getContext("2d"), {
        type: "line",
        data: { labels: labels, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },  // crosshair behaviour
          layout: { padding: { right: 10, top: 4 } },
          plugins: {
            legend: { display: false },     // custom HTML legend below
            tooltip: Object.assign(tooltipStyle(t), {
              displayColors: true,
              callbacks: {
                title: function (items) {
                  return new Date(items[0].label).toLocaleDateString(undefined,
                    { year: "numeric", month: "short", day: "numeric" });
                },
                label: function (item) {
                  var v = item.parsed.y;
                  return item.dataset.label + "  " +
                         (v >= 100 ? "+" : "") + (v - 100).toFixed(1) + "%";
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
                autoSkip: true, maxTicksLimit: 7,
                callback: function (val, idx) {
                  var d = new Date(this.getLabelForValue(val));
                  return d.toLocaleDateString(undefined,
                    { month: "short", day: "numeric" });
                }
              }
            },
            y: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: {
                color: t.muted, font: { size: 11.5 }, padding: 8,
                callback: function (v) { return v.toFixed(0); }
              }
            }
          },
          animation: { duration: 420 }
        }
      });

      if (trendLegend) {
        trendLegend.innerHTML = symbols.map(function (sym, i) {
          return '<span class="legend-item">' +
                 '<span class="legend-key" style="background:' +
                 t.series[i % t.series.length] + '"></span>' + sym + "</span>";
        }).join("");
      }
    };

    register(drawTrend);
    drawTrend();
  }

  // --------------------------------------------------- single-price chart --
  var seriesData = readJSON("series-data");
  var priceCanvas = document.getElementById("price-chart");

  if (seriesData && priceCanvas) {
    var pChart = null;

    var drawPrice = function () {
      var t = tokens();
      if (pChart) pChart.destroy();

      var up = seriesData.closes[seriesData.closes.length - 1] >= seriesData.closes[0];
      var line = up ? t.pos : t.neg;

      var ctx = priceCanvas.getContext("2d");
      var fill = ctx.createLinearGradient(0, 0, 0, priceCanvas.clientHeight || 380);
      fill.addColorStop(0, line + "2e");
      fill.addColorStop(1, line + "00");

      pChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: seriesData.dates,
          datasets: [{
            label: seriesData.symbol,
            data: seriesData.closes,
            borderColor: line,
            backgroundColor: fill,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: t.surface,
            pointHoverBackgroundColor: line,
            fill: true,
            tension: 0.15
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          layout: { padding: { right: 12, top: 4 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tooltipStyle(t), {
              callbacks: {
                title: function (items) {
                  return new Date(items[0].label).toLocaleDateString(undefined,
                    { year: "numeric", month: "short", day: "numeric" });
                },
                label: function (item) {
                  var i = item.dataIndex;
                  return ["Close:  " + money(item.parsed.y),
                          "Volume: " + volume(seriesData.volumes[i] || 0)];
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
                callback: function (val) {
                  var d = new Date(this.getLabelForValue(val));
                  return d.toLocaleDateString(undefined,
                    { month: "short", day: "numeric" });
                }
              }
            },
            y: {
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: {
                color: t.muted, font: { size: 11.5 }, padding: 8,
                callback: function (v) { return "$" + v.toLocaleString(); }
              }
            }
          },
          animation: { duration: 420 }
        }
      });
    };

    register(drawPrice);
    drawPrice();
  }

  // ------------------------------------------------------ drawdown chart ---
  var ddData = readJSON("drawdown-data");
  var ddCanvas = document.getElementById("dd-chart");

  if (ddData && ddCanvas) {
    var ddChart = null;

    var drawDd = function () {
      var t = tokens();
      if (ddChart) ddChart.destroy();

      var ctx = ddCanvas.getContext("2d");
      var fill = ctx.createLinearGradient(0, 0, 0, ddCanvas.clientHeight || 300);
      // Drawdown is always negative, so it is shaded down from zero in the
      // loss colour - the shape is the story.
      fill.addColorStop(0, t.neg + "00");
      fill.addColorStop(1, t.neg + "3a");

      ddChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: ddData.dates,
          datasets: [{
            label: "Drawdown",
            data: ddData.values,
            borderColor: t.neg,
            backgroundColor: fill,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: t.surface,
            fill: true,
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          layout: { padding: { right: 12, top: 4 } },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tooltipStyle(t), {
              callbacks: {
                title: function (items) {
                  return new Date(items[0].label).toLocaleDateString(undefined,
                    { year: "numeric", month: "short", day: "numeric" });
                },
                label: function (item) {
                  return item.parsed.y.toFixed(1) + "% below its high";
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
              max: 0,
              grid: { color: t.grid, drawTicks: false },
              border: { display: false },
              ticks: { color: t.muted, font: { size: 11.5 }, padding: 8,
                       callback: function (v) { return v + "%"; } }
            }
          },
          animation: { duration: 420 }
        }
      });
    };

    register(drawDd);
    drawDd();
  }

  // ------------------------------------------------------------- refresh ---
  var refreshBtn = document.getElementById("refresh-btn");
  var refreshStatus = document.getElementById("refresh-status");

  function showStatus(msg) {
    if (!refreshStatus) return;
    refreshStatus.hidden = false;
    refreshStatus.textContent = msg;
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", function () {
      refreshBtn.disabled = true;
      showStatus("Starting refresh...");

      fetch("/api/refresh", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (j) {
          showStatus(j.message || "Refreshing...");
          var poll = setInterval(function () {
            fetch("/api/refresh/status")
              .then(function (r) { return r.json(); })
              .then(function (s) {
                showStatus(s.message);
                if (!s.running) {
                  clearInterval(poll);
                  refreshBtn.disabled = false;
                  showStatus(s.message + " - reloading...");
                  setTimeout(function () { location.reload(); }, 900);
                }
              });
          }, 1500);
        })
        .catch(function () {
          showStatus("Could not start the refresh.");
          refreshBtn.disabled = false;
        });
    });
  }
})();
