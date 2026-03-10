// ═══════════════════════════════════════════════
// CardioTwin Results Dashboard — results.js
// Fully responsive via ResizeObserver
// ═══════════════════════════════════════════════
(function () {
  const MODELS = [
    { key: "baseline", label: "CardioTwin", short: "CT", color: "#4f8ef7" },
    { key: "nogate_bce", label: "No-Gate BCE", short: "NG", color: "#f472b6" },
    {
      key: "baseline_bce_weighted",
      label: "BCE Weighted",
      short: "BCE-W",
      color: "#34d399",
    },
    {
      key: "baseline_focal",
      label: "Focal Loss",
      short: "Focal",
      color: "#fb923c",
    },
    {
      key: "mlp_embedding",
      label: "MLP + Embed",
      short: "MLP+",
      color: "#a78bfa",
    },
    {
      key: "mlp_baseline",
      label: "MLP Baseline",
      short: "MLP",
      color: "#38bdf8",
    },
    {
      key: "xgboost_embedding",
      label: "XGBoost + Embed",
      short: "XG+",
      color: "#2dd4bf",
    },
    {
      key: "xgboost_baseline",
      label: "XGBoost Baseline",
      short: "XG",
      color: "#fbbf24",
    },
  ];
  const MODEL_KEYS = MODELS.map((m) => m.key);
  const modelOf = (key) => MODELS.find((m) => m.key === key);
  const colorOf = (key) => modelOf(key)?.color ?? "#aaa";

  const TARGET_LABELS = {
    hypertension_crisis: "Hypertension Crisis",
    afib_aflutter: "AFib / Aflutter",
    chronic_ischemic_disease: "Chronic Ischemic",
    heart_failure_acute: "HF Acute",
    brady_heart_block_conduction: "Brady / Block",
    stroke_tia: "Stroke / TIA",
    valvular_endocardial_disease: "Valvular",
    heart_failure_chronic: "HF Chronic",
    ventricular_arrhythmias_arrest: "Ventricular Arrhy.",
    cardiomyopathy_myocarditis: "Cardiomyopathy",
    ami_nstemi: "AMI NSTEMI",
    ami_stemi: "AMI STEMI",
    supraventricular_tachyarrhythmias: "SVT",
    aortic_peripheral_vascular: "Aortic / PVD",
    pe_dvt_venous_thromboembolism: "PE / DVT",
    unstable_angina_ac_ischemia: "Unstable Angina",
    pericardial_disease_tamponade: "Pericardial",
  };

  const CSV_OVERALL = "misc/all_models_overall.csv";
  const CSV_PER_TARGET = "misc/all_models_per_target.csv";

  function parseCSV(text) {
    const lines = text.trim().split("\n");
    const headers = lines[0].split(",");
    return lines.slice(1).map((line) => {
      const vals = line.split(",");
      const obj = {};
      headers.forEach((h, i) => {
        obj[h] = isNaN(+vals[i]) ? vals[i] : +vals[i];
      });
      return obj;
    });
  }

  function setLoading(msg) {
    ["res-bar-chart", "res-radar-chart", "res-heatmap"].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.setAttribute("viewBox", "0 0 400 120");
      el.innerHTML = `<text x="200" y="60" text-anchor="middle" dominant-baseline="middle"
        fill="#86868b" font-size="14" font-family="-apple-system,sans-serif">${msg}</text>`;
    });
  }

  function clearSVG(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
  }

  setLoading("Loading data\u2026");

  Promise.all([
    fetch(CSV_OVERALL).then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.text();
    }),
    fetch(CSV_PER_TARGET).then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.text();
    }),
  ])
    .then(([overallText, perTargetText]) => {
      const overall = parseCSV(overallText).filter((d) =>
        MODEL_KEYS.includes(d.model_type),
      );
      const perTarget = parseCSV(perTargetText).filter((d) =>
        MODEL_KEYS.includes(d.model_type),
      );
      const TARGETS = [...new Set(perTarget.map((d) => d.target))];
      clearSVG("res-bar-chart");
      clearSVG("res-radar-chart");
      clearSVG("res-heatmap");
      init(overall, perTarget, TARGETS);
    })
    .catch((err) => {
      setLoading("Failed to load CSV (" + err.message + ")");
      console.error("CardioTwin dashboard: CSV fetch error", err);
    });

  function init(overall, perTarget, TARGETS) {
    // TOOLTIP
    const tip = document.createElement("div");
    tip.className = "res-tooltip";
    document.body.appendChild(tip);

    function showTip(html, e) {
      tip.innerHTML = html;
      tip.classList.add("res-tip-visible");
      moveTip(e);
    }
    function moveTip(e) {
      tip.style.left = Math.min(e.clientX + 16, window.innerWidth - 265) + "px";
      tip.style.top = Math.min(e.clientY - 10, window.innerHeight - 220) + "px";
    }
    function hideTip() {
      tip.classList.remove("res-tip-visible");
    }

    function makeTip(key, row, subtitle) {
      const m = modelOf(key);
      return (
        '<div class="res-tip-title" style="color:' +
        m.color +
        '">' +
        m.label +
        "</div>" +
        (subtitle
          ? '<div style="color:#86868b;font-size:11px;margin-bottom:5px">' +
            subtitle +
            "</div>"
          : "") +
        '<div class="res-tip-row"><span class="res-tip-label">ROC-AUC</span><span class="res-tip-val">' +
        (row.mean_roc_auc ?? row.roc_auc ?? 0).toFixed(3) +
        "</span></div>" +
        '<div class="res-tip-row"><span class="res-tip-label">PR-AUC</span><span class="res-tip-val">' +
        (row.mean_pr_auc ?? row.pr_auc ?? 0).toFixed(3) +
        "</span></div>" +
        '<div class="res-tip-row"><span class="res-tip-label">F1</span><span class="res-tip-val">' +
        (row.f1 ?? 0).toFixed(3) +
        "</span></div>" +
        '<div class="res-tip-row"><span class="res-tip-label">Precision</span><span class="res-tip-val">' +
        (row.precision ?? 0).toFixed(3) +
        "</span></div>" +
        '<div class="res-tip-row"><span class="res-tip-label">Recall</span><span class="res-tip-val">' +
        (row.recall ?? 0).toFixed(3) +
        "</span></div>"
      );
    }

    // HOVER STATE
    let hovered = null;
    function applyHover() {
      d3.selectAll(".res-bar-rect").classed("res-dimmed", function (d) {
        return hovered && d.model !== hovered;
      });
      d3.selectAll(".res-radar-poly").classed("res-dimmed", function (d) {
        return hovered && d.key !== hovered;
      });
      d3.selectAll(".res-hm-cell").classed("res-dimmed", function (d) {
        return hovered && d.model !== hovered;
      });
      d3.selectAll(".res-hm-val").classed("res-dimmed", function (d) {
        return hovered && d.model !== hovered;
      });
      document.querySelectorAll(".res-legend-item").forEach(function (el) {
        el.classList.toggle("res-legend-active", el.dataset.key === hovered);
      });
    }

    // LEGEND
    var legendEl = document.getElementById("res-legend");
    MODELS.forEach(function (m) {
      var item = document.createElement("div");
      item.className = "res-legend-item";
      item.dataset.key = m.key;
      item.style.setProperty("--res-model-color", m.color);
      item.innerHTML =
        '<span class="res-legend-dot" style="background:' +
        m.color +
        '"></span>' +
        m.label;
      item.addEventListener("mouseenter", function () {
        hovered = m.key;
        applyHover();
      });
      item.addEventListener("mouseleave", function () {
        hovered = null;
        applyHover();
      });
      legendEl.appendChild(item);
    });

    // ════════════════════════════════
    // PLOT 1 — BAR CHART
    // ════════════════════════════════
    var METRICS = [
      { key: "mean_roc_auc", label: "ROC-AUC", domain: [0.83, 0.935] },
      { key: "f1", label: "F1", domain: [0, 0.58] },
      { key: "precision", label: "Precision", domain: [0, 0.86] },
      { key: "recall", label: "Recall", domain: [0, 0.6] },
    ];
    var activeMetric = METRICS[0];

    var toggleRow = document.getElementById("metric-toggles");
    METRICS.forEach(function (m) {
      var btn = document.createElement("button");
      btn.className =
        "res-toggle-btn" + (m === activeMetric ? " res-toggle-active" : "");
      btn.textContent = m.label;
      btn.addEventListener("click", function () {
        activeMetric = m;
        document.querySelectorAll(".res-toggle-btn").forEach(function (b) {
          b.classList.remove("res-toggle-active");
        });
        btn.classList.add("res-toggle-active");
        drawBar();
      });
      toggleRow.appendChild(btn);
    });

    var barSvgEl = document.getElementById("res-bar-chart");

    function drawBar() {
      var container = barSvgEl.parentElement;
      var totalW =
        Math.floor(container.getBoundingClientRect().width) ||
        container.clientWidth ||
        400;
      if (totalW < 10) return;
      var isMobile = totalW < 520;
      var bM = {
        top: 16,
        right: 16,
        bottom: isMobile ? 48 : 56,
        left: isMobile ? 42 : 52,
      };
      var bW = totalW - bM.left - bM.right;
      var bH = isMobile ? 180 : 220;

      barSvgEl.innerHTML = "";
      var bSvg = d3
        .select(barSvgEl)
        .attr("viewBox", "0 0 " + totalW + " " + (bH + bM.top + bM.bottom))
        .attr("width", "100%")
        .attr("height", bH + bM.top + bM.bottom);

      var bG = bSvg
        .append("g")
        .attr("transform", "translate(" + bM.left + "," + bM.top + ")");
      var bX = d3.scaleBand().domain(MODEL_KEYS).range([0, bW]).padding(0.3);
      var bY = d3
        .scaleLinear()
        .domain(activeMetric.domain)
        .nice()
        .range([bH, 0]);

      bY.ticks(4).forEach(function (t) {
        bG.append("line")
          .attr("x1", 0)
          .attr("x2", bW)
          .attr("y1", bY(t))
          .attr("y2", bY(t))
          .attr("stroke", "#e5e5ea")
          .attr("stroke-width", 0.5);
      });

      bG.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + bH + ")")
        .call(
          d3
            .axisBottom(bX)
            .tickFormat(function (k) {
              return modelOf(k) ? modelOf(k).short : k;
            })
            .tickSize(0),
        )
        .selectAll("text")
        .attr("dy", "1.2em")
        .style("font-size", isMobile ? "10px" : "13px")
        .style("font-family", "JetBrains Mono, monospace");

      bG.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(bY).ticks(4).tickFormat(d3.format(".2f")).tickSize(0))
        .selectAll("text")
        .attr("dx", "-0.4em")
        .style("font-size", isMobile ? "10px" : "13px");

      MODEL_KEYS.forEach(function (mk) {
        var row = overall.find(function (r) {
          return r.model_type === mk;
        });
        if (!row) return;
        var val = row[activeMetric.key];
        var x = bX(mk),
          w = bX.bandwidth();
        var datum = { model: mk, val: val, row: row };

        bG.append("rect")
          .attr("class", "res-bar-rect")
          .datum(datum)
          .attr("x", x)
          .attr("width", w)
          .attr("y", bH)
          .attr("height", 0)
          .attr("rx", 3)
          .attr("fill", colorOf(mk))
          .on("mouseover", function (ev) {
            hovered = mk;
            applyHover();
            showTip(makeTip(mk, row), ev);
          })
          .on("mousemove", moveTip)
          .on("mouseout", function () {
            hovered = null;
            applyHover();
            hideTip();
          })
          .transition()
          .duration(420)
          .ease(d3.easeCubicOut)
          .attr("y", bY(val))
          .attr("height", bH - bY(val));

        if (!isMobile) {
          bG.append("text")
            .datum(datum)
            .attr("class", "res-bar-label")
            .attr("x", x + w / 2)
            .attr("text-anchor", "middle")
            .attr("font-size", "11px")
            .attr("font-family", "JetBrains Mono, monospace")
            .attr("fill", colorOf(mk))
            .attr("pointer-events", "none")
            .transition()
            .duration(420)
            .ease(d3.easeCubicOut)
            .attr("y", bY(val) - 4)
            .text(val.toFixed(3));
        }
      });

      applyHover();
    }

    drawBar();
    new ResizeObserver(function () {
      requestAnimationFrame(drawBar);
    }).observe(barSvgEl.parentElement);

    // ════════════════════════════════
    // PLOT 2 — RADAR
    // ════════════════════════════════
    var RAXES = [
      { key: "mean_roc_auc", label: "ROC-AUC", min: 0.83, max: 0.935 },
      { key: "mean_pr_auc", label: "PR-AUC", min: 0.17, max: 0.28 },
      { key: "f1", label: "F1", min: 0.0, max: 0.56 },
      { key: "recall", label: "Recall", min: 0.0, max: 0.6 },
      { key: "precision", label: "Precision", min: 0.0, max: 0.86 },
    ];
    var RN = RAXES.length;
    var radarSvgEl = document.getElementById("res-radar-chart");

    function drawRadar() {
      var container = radarSvgEl.parentElement;
      var totalW =
        Math.floor(container.getBoundingClientRect().width) ||
        container.clientWidth ||
        360;
      if (totalW < 10) return;
      var isMobile = totalW < 400;
      // Padding around the radar circle for axis labels
      var labelPad = isMobile ? 36 : 48;
      var rW = totalW;
      // Height gives room for labels top + bottom
      var rH = totalW + labelPad * 0.5;
      var rR = Math.min(rW, rH) / 2 - labelPad - (isMobile ? 4 : 8);
      var rcx = rW / 2;
      var rcy = rH / 2;
      var labelOffset = rR + (isMobile ? 18 : 24);
      var fontSize = isMobile ? "10px" : "12px";

      radarSvgEl.innerHTML = "";
      var rSvg = d3
        .select(radarSvgEl)
        .attr("viewBox", "0 0 " + rW + " " + rH)
        .attr("width", "100%")
        .attr("height", rH);
      var rG = rSvg.append("g");

      [0.25, 0.5, 0.75, 1].forEach(function (t) {
        var pts = RAXES.map(function (_, i) {
          var a = (i / RN) * 2 * Math.PI - Math.PI / 2;
          return [rcx + rR * t * Math.cos(a), rcy + rR * t * Math.sin(a)];
        });
        rG.append("polygon")
          .attr(
            "points",
            pts
              .map(function (p) {
                return p.join(",");
              })
              .join(" "),
          )
          .attr("fill", "none")
          .attr("stroke", "#e5e5ea")
          .attr("stroke-width", 0.5);
      });

      RAXES.forEach(function (ax, i) {
        var a = (i / RN) * 2 * Math.PI - Math.PI / 2;
        rG.append("line")
          .attr("x1", rcx)
          .attr("y1", rcy)
          .attr("x2", rcx + rR * Math.cos(a))
          .attr("y2", rcy + rR * Math.sin(a))
          .attr("stroke", "#e5e5ea")
          .attr("stroke-width", 0.5);
        rG.append("text")
          .attr("x", rcx + labelOffset * Math.cos(a))
          .attr("y", rcy + labelOffset * Math.sin(a))
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", "#333")
          .attr("font-size", fontSize)
          .attr("font-weight", "500")
          .attr("font-family", "-apple-system,BlinkMacSystemFont,sans-serif")
          .text(ax.label);
      });

      MODELS.forEach(function (m) {
        var row = overall.find(function (r) {
          return r.model_type === m.key;
        });
        if (!row) return;
        var pts = RAXES.map(function (ax, i) {
          var t = Math.max(
            0,
            Math.min(1, (row[ax.key] - ax.min) / (ax.max - ax.min)),
          );
          var a = (i / RN) * 2 * Math.PI - Math.PI / 2;
          return [rcx + rR * t * Math.cos(a), rcy + rR * t * Math.sin(a)];
        });
        rG.append("polygon")
          .datum({ key: m.key })
          .attr("class", "res-radar-poly")
          .attr(
            "points",
            pts
              .map(function (p) {
                return p.join(",");
              })
              .join(" "),
          )
          .attr("fill", m.color)
          .attr("fill-opacity", 0.1)
          .attr("stroke", m.color)
          .attr("stroke-width", 2)
          .style("cursor", "pointer")
          .on("mouseover", function (ev) {
            hovered = m.key;
            applyHover();
            showTip(makeTip(m.key, row), ev);
          })
          .on("mousemove", moveTip)
          .on("mouseout", function () {
            hovered = null;
            applyHover();
            hideTip();
          });
      });

      applyHover();
    }

    drawRadar();
    new ResizeObserver(function () {
      requestAnimationFrame(drawRadar);
    }).observe(radarSvgEl.parentElement);

    // ════════════════════════════════
    // PLOT 3 — HEATMAP
    // ════════════════════════════════
    var hmSvgEl = document.getElementById("res-heatmap");

    function drawHeatmap() {
      var container = hmSvgEl.closest(".res-heatmap-wrap");
      var availW =
        Math.floor(container.getBoundingClientRect().width) ||
        container.clientWidth ||
        320;
      if (availW < 10) return;
      var isMobile = availW < 500;

      // Compute natural cell sizes
      var cellWNatural = isMobile ? 36 : 54;
      var cellH = isMobile ? 28 : 38;
      var rowLabelW = isMobile ? 44 : 80;
      var colLabelH = isMobile ? 76 : 106;
      var hmM = { top: 20, right: 16, bottom: colLabelH + 24, left: rowLabelW };

      // Natural content width
      var hmWNatural = TARGETS.length * cellWNatural;
      var totalWNatural = hmWNatural + hmM.left + hmM.right;

      // If content fits, stretch cells to fill container; otherwise use natural size and scroll
      var fits = totalWNatural <= availW;
      var cellW = fits
        ? Math.floor((availW - hmM.left - hmM.right) / TARGETS.length)
        : cellWNatural;
      var hmW = TARGETS.length * cellW;
      var hmH = MODEL_KEYS.length * cellH;
      var totalW = hmW + hmM.left + hmM.right;
      var totalH = hmH + hmM.top + hmM.bottom;

      hmSvgEl.innerHTML = "";
      d3.select(hmSvgEl)
        .attr("viewBox", "0 0 " + totalW + " " + totalH)
        .attr("width", totalW)
        .attr("height", totalH)
        .style("min-width", totalW + "px")
        .style("display", "block");

      var hmG = d3
        .select(hmSvgEl)
        .append("g")
        .attr("transform", "translate(" + hmM.left + "," + hmM.top + ")");

      var hmColor = d3
        .scaleLinear()
        .domain([0.74, 0.855, 0.96])
        .range(["#fef3c7", "#ffffff", "#3b82f6"])
        .clamp(true);

      var cellData = [];
      MODEL_KEYS.forEach(function (mk) {
        TARGETS.forEach(function (t) {
          var row = perTarget.find(function (r) {
            return r.model_type === mk && r.target === t;
          });
          cellData.push({ model: mk, target: t, row: row || null });
        });
      });

      hmG
        .selectAll(".res-hm-cell")
        .data(cellData)
        .enter()
        .append("rect")
        .attr("class", "res-hm-cell")
        .attr("x", function (d) {
          return TARGETS.indexOf(d.target) * cellW + 1;
        })
        .attr("y", function (d) {
          return MODEL_KEYS.indexOf(d.model) * cellH + 1;
        })
        .attr("width", cellW - 2)
        .attr("height", cellH - 2)
        .attr("rx", 3)
        .attr("fill", function (d) {
          return d.row ? hmColor(d.row.roc_auc) : "#f5f5f7";
        })
        .style("cursor", "pointer")
        .on("mouseover", function (ev, d) {
          if (!d.row) return;
          hovered = d.model;
          applyHover();
          showTip(
            makeTip(d.model, d.row, TARGET_LABELS[d.target] || d.target),
            ev,
          );
        })
        .on("mousemove", moveTip)
        .on("mouseout", function () {
          hovered = null;
          applyHover();
          hideTip();
        })
        .on("click", function (_, d) {
          if (d.row) showDetailCell(d);
        });

      if (!isMobile) {
        hmG
          .selectAll(".res-hm-val")
          .data(cellData)
          .enter()
          .append("text")
          .attr("class", "res-hm-val")
          .attr("x", function (d) {
            return TARGETS.indexOf(d.target) * cellW + cellW / 2;
          })
          .attr("y", function (d) {
            return MODEL_KEYS.indexOf(d.model) * cellH + cellH / 2 + 1;
          })
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "10px")
          .attr("font-weight", "600")
          .attr("font-family", "JetBrains Mono, monospace")
          .attr("fill", function (d) {
            return !d.row ? "#aaa" : d.row.roc_auc > 0.91 ? "#fff" : "#1d1d1f";
          })
          .attr("pointer-events", "none")
          .text(function (d) {
            return d.row ? d.row.roc_auc.toFixed(2) : "\u2014";
          });
      }

      MODEL_KEYS.forEach(function (mk, i) {
        var m = modelOf(mk);
        hmG
          .append("text")
          .attr("x", -6)
          .attr("y", i * cellH + cellH / 2 + 1)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", isMobile ? "9px" : "11px")
          .attr("font-family", "JetBrains Mono, monospace")
          .attr("font-weight", "500")
          .attr("fill", m.color)
          .style("cursor", "pointer")
          .text(m.short)
          .on("mouseover", function () {
            hovered = mk;
            applyHover();
          })
          .on("mouseout", function () {
            hovered = null;
            applyHover();
          });
      });

      var colFontSize = isMobile ? "9px" : "11px";
      var colAngle = isMobile ? 50 : 40;
      TARGETS.forEach(function (t, i) {
        hmG
          .append("text")
          .attr(
            "transform",
            "translate(" +
              (i * cellW + cellW / 2) +
              "," +
              (hmH + 8) +
              ") rotate(" +
              colAngle +
              ")",
          )
          .attr("text-anchor", "start")
          .attr("font-size", colFontSize)
          .attr("font-family", "-apple-system,sans-serif")
          .attr("fill", "#444")
          .style("cursor", "pointer")
          .text(TARGET_LABELS[t] || t)
          .on("click", function () {
            showDetailColumn(t);
          });
      });

      var defs = d3.select(hmSvgEl).append("defs");
      var gid = "hmGradRes4";
      var lg = defs.append("linearGradient").attr("id", gid);
      lg.append("stop").attr("offset", "0%").attr("stop-color", "#fef3c7");
      lg.append("stop").attr("offset", "50%").attr("stop-color", "#ffffff");
      lg.append("stop").attr("offset", "100%").attr("stop-color", "#3b82f6");

      var lgG = d3
        .select(hmSvgEl)
        .append("g")
        .attr(
          "transform",
          "translate(" + hmM.left + "," + (hmM.top + hmH + colLabelH + 4) + ")",
        );
      lgG
        .append("rect")
        .attr("width", 120)
        .attr("height", 7)
        .attr("rx", 4)
        .attr("fill", "url(#" + gid + ")")
        .attr("stroke", "#e5e5ea")
        .attr("stroke-width", 0.5);
      [
        ["0", "0.74"],
        ["60", "0.855"],
        ["120", "0.96 \u2192"],
      ].forEach(function (pair) {
        lgG
          .append("text")
          .attr("x", +pair[0])
          .attr("y", -5)
          .attr(
            "text-anchor",
            +pair[0] === 120 ? "end" : +pair[0] === 60 ? "middle" : "start",
          )
          .attr("font-size", "9px")
          .attr("font-family", "JetBrains Mono,monospace")
          .attr("fill", "#86868b")
          .text(pair[1]);
      });
      lgG
        .append("text")
        .attr("x", 0)
        .attr("y", 18)
        .attr("font-size", "9px")
        .attr("font-family", "-apple-system,sans-serif")
        .attr("fill", "#86868b")
        .text("ROC-AUC");

      applyHover();
    }

    drawHeatmap();
    new ResizeObserver(function () {
      requestAnimationFrame(drawHeatmap);
    }).observe(hmSvgEl.closest(".res-heatmap-wrap"));

    // ── DETAIL PANEL ────────────────────────────────
    var detailEl = document.getElementById("res-detail");
    var DBARS = [
      { key: "roc_auc", label: "ROC-AUC", color: "#4f8ef7", max: 1 },
      { key: "pr_auc", label: "PR-AUC", color: "#a78bfa", max: 0.85 },
      { key: "f1", label: "F1", color: "#34d399", max: 1 },
      { key: "precision", label: "Precision", color: "#fb923c", max: 1 },
      { key: "recall", label: "Recall", color: "#f472b6", max: 1 },
    ];

    function showDetailCell(d) {
      var r = d.row,
        m = modelOf(d.model);
      detailEl.innerHTML =
        '<div class="res-detail-title">' +
        '<span style="color:' +
        m.color +
        '">' +
        m.label +
        "</span>" +
        '<span style="color:#86868b;font-weight:400"> \u00b7 ' +
        (TARGET_LABELS[d.target] || d.target) +
        "</span>" +
        '</div><div class="res-detail-bars">' +
        DBARS.map(function (db) {
          var val = r[db.key] || 0,
            pct = Math.min(100, (val / db.max) * 100);
          return (
            '<div class="res-dbar-row">' +
            '<div class="res-dbar-label">' +
            db.label +
            "</div>" +
            '<div class="res-dbar-track"><div class="res-dbar-fill" style="width:' +
            pct +
            "%;background:" +
            db.color +
            '"></div></div>' +
            '<div class="res-dbar-val">' +
            val.toFixed(3) +
            "</div>" +
            "</div>"
          );
        }).join("") +
        "</div>";
    }

    function showDetailColumn(target) {
      var label = TARGET_LABELS[target] || target;
      var rows = MODEL_KEYS.map(function (mk) {
        return {
          model: mk,
          r: perTarget.find(function (x) {
            return x.model_type === mk && x.target === target;
          }),
        };
      })
        .filter(function (x) {
          return x.r;
        })
        .sort(function (a, b) {
          return b.r.roc_auc - a.r.roc_auc;
        });

      detailEl.innerHTML =
        '<div class="res-detail-title">' +
        label +
        '<span style="color:#86868b;font-weight:400"> \u00b7 ROC-AUC ranking</span>' +
        '</div><div class="res-detail-bars">' +
        rows
          .map(function (x, i) {
            var m = modelOf(x.model);
            var pct = Math.min(
              100,
              ((x.r.roc_auc - 0.74) / (0.96 - 0.74)) * 100,
            );
            return (
              '<div class="res-dbar-row">' +
              '<div class="res-dbar-label" style="color:' +
              m.color +
              '">#' +
              (i + 1) +
              " " +
              m.short +
              "</div>" +
              '<div class="res-dbar-track"><div class="res-dbar-fill" style="width:' +
              pct +
              "%;background:" +
              m.color +
              '"></div></div>' +
              '<div class="res-dbar-val">' +
              x.r.roc_auc.toFixed(3) +
              "</div>" +
              "</div>"
            );
          })
          .join("") +
        "</div>";
    }
  } // end init()
})();
