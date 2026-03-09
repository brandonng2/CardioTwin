// ═══════════════════════════════════════════════
// CardioTwin Results Dashboard — results.js
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

  // show spinner text in each SVG before data loads
  function setLoading(msg) {
    ["res-bar-chart", "res-radar-chart", "res-heatmap"].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.setAttribute("viewBox", "0 0 400 120");
      el.innerHTML = `<text x="200" y="60" text-anchor="middle" dominant-baseline="middle"
        fill="#86868b" font-size="14" font-family="-apple-system,sans-serif">${msg}</text>`;
    });
  }

  // completely wipe SVG content so loading text doesn't persist
  function clearSVG(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
  }

  setLoading("Loading data…");

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

      // clear loading placeholders before drawing
      clearSVG("res-bar-chart");
      clearSVG("res-radar-chart");
      clearSVG("res-heatmap");

      init(overall, perTarget, TARGETS);
    })
    .catch((err) => {
      setLoading(`Failed to load CSV (${err.message})`);
      console.error("CardioTwin dashboard: CSV fetch error", err);
    });

  // ── MAIN INIT ────────────────────────────────────
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
      return `
        <div class="res-tip-title" style="color:${m.color}">${m.label}</div>
        ${subtitle ? `<div style="color:#86868b;font-size:11px;margin-bottom:5px">${subtitle}</div>` : ""}
        <div class="res-tip-row"><span class="res-tip-label">ROC-AUC</span><span class="res-tip-val">${(row.mean_roc_auc ?? row.roc_auc ?? 0).toFixed(3)}</span></div>
        <div class="res-tip-row"><span class="res-tip-label">PR-AUC</span><span class="res-tip-val">${(row.mean_pr_auc ?? row.pr_auc ?? 0).toFixed(3)}</span></div>
        <div class="res-tip-row"><span class="res-tip-label">F1</span><span class="res-tip-val">${(row.f1 ?? 0).toFixed(3)}</span></div>
        <div class="res-tip-row"><span class="res-tip-label">Precision</span><span class="res-tip-val">${(row.precision ?? 0).toFixed(3)}</span></div>
        <div class="res-tip-row"><span class="res-tip-label">Recall</span><span class="res-tip-val">${(row.recall ?? 0).toFixed(3)}</span></div>`;
    }

    // HOVER STATE
    let hovered = null;
    function applyHover() {
      d3.selectAll(".res-bar-rect").classed(
        "res-dimmed",
        (d) => hovered && d.model !== hovered,
      );
      d3.selectAll(".res-radar-poly").classed(
        "res-dimmed",
        (d) => hovered && d.key !== hovered,
      );
      d3.selectAll(".res-hm-cell").classed(
        "res-dimmed",
        (d) => hovered && d.model !== hovered,
      );
      d3.selectAll(".res-hm-val").classed(
        "res-dimmed",
        (d) => hovered && d.model !== hovered,
      );
      document.querySelectorAll(".res-legend-item").forEach((el) => {
        el.classList.toggle("res-legend-active", el.dataset.key === hovered);
      });
    }

    // LEGEND
    const legendEl = document.getElementById("res-legend");
    MODELS.forEach((m) => {
      const item = document.createElement("div");
      item.className = "res-legend-item";
      item.dataset.key = m.key;
      item.style.setProperty("--res-model-color", m.color);
      item.innerHTML = `<span class="res-legend-dot" style="background:${m.color}"></span>${m.label}`;
      item.addEventListener("mouseenter", () => {
        hovered = m.key;
        applyHover();
      });
      item.addEventListener("mouseleave", () => {
        hovered = null;
        applyHover();
      });
      legendEl.appendChild(item);
    });

    // ════════════════════════════════
    // PLOT 1 — BAR CHART
    // ════════════════════════════════
    const METRICS = [
      { key: "mean_roc_auc", label: "ROC-AUC", domain: [0.83, 0.935] },
      { key: "f1", label: "F1", domain: [0, 0.58] },
      { key: "precision", label: "Precision", domain: [0, 0.86] },
      { key: "recall", label: "Recall", domain: [0, 0.6] },
    ];
    let activeMetric = METRICS[0];

    const toggleRow = document.getElementById("metric-toggles");
    METRICS.forEach((m) => {
      const btn = document.createElement("button");
      btn.className =
        "res-toggle-btn" + (m === activeMetric ? " res-toggle-active" : "");
      btn.textContent = m.label;
      btn.addEventListener("click", () => {
        activeMetric = m;
        document
          .querySelectorAll(".res-toggle-btn")
          .forEach((b) => b.classList.remove("res-toggle-active"));
        btn.classList.add("res-toggle-active");
        updateBars();
      });
      toggleRow.appendChild(btn);
    });

    const bM = { top: 12, right: 16, bottom: 60, left: 52 };
    const bW = 580,
      bH = 240;
    const bSvg = d3
      .select("#res-bar-chart")
      .attr(
        "viewBox",
        `0 0 ${bW + bM.left + bM.right} ${bH + bM.top + bM.bottom}`,
      )
      .attr("width", "100%");
    const bG = bSvg
      .append("g")
      .attr("transform", `translate(${bM.left},${bM.top})`);
    const bX = d3.scaleBand().domain(MODEL_KEYS).range([0, bW]).padding(0.3);
    const bY = d3.scaleLinear().range([bH, 0]);
    const bXA = bG
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${bH})`);
    const bYA = bG.append("g").attr("class", "axis");
    const bGr = bG.append("g");

    function updateBars() {
      bY.domain(activeMetric.domain).nice();
      bGr.selectAll("line").remove();
      bY.ticks(4).forEach((t) =>
        bGr
          .append("line")
          .attr("x1", 0)
          .attr("x2", bW)
          .attr("y1", bY(t))
          .attr("y2", bY(t))
          .attr("stroke", "#e5e5ea")
          .attr("stroke-width", 0.5),
      );
      bXA
        .call(
          d3
            .axisBottom(bX)
            .tickFormat((k) => modelOf(k)?.short ?? k)
            .tickSize(0),
        )
        .selectAll("text")
        .attr("dy", "1.4em")
        .style("font-size", "15px")
        .style("font-family", "JetBrains Mono, monospace");
      bYA
        .call(d3.axisLeft(bY).ticks(4).tickFormat(d3.format(".2f")).tickSize(0))
        .selectAll("text")
        .attr("dx", "-0.6em")
        .style("font-size", "14px");

      const groups = bG.selectAll(".res-bar-group").data(MODEL_KEYS);
      const allG = groups
        .enter()
        .append("g")
        .attr("class", "res-bar-group")
        .merge(groups)
        .attr("transform", (d) => `translate(${bX(d)},0)`);

      const bars = allG.selectAll(".res-bar-rect").data((d) => {
        const row = overall.find((r) => r.model_type === d);
        return row ? [{ model: d, val: row[activeMetric.key], row }] : [];
      });

      bars
        .enter()
        .append("rect")
        .attr("class", "res-bar-rect")
        .attr("x", 0)
        .attr("width", bX.bandwidth())
        .attr("y", bH)
        .attr("height", 0)
        .attr("rx", 4)
        .attr("fill", (d) => colorOf(d.model))
        .merge(bars)
        .on("mouseover", (ev, d) => {
          hovered = d.model;
          applyHover();
          showTip(makeTip(d.model, d.row), ev);
        })
        .on("mousemove", moveTip)
        .on("mouseout", () => {
          hovered = null;
          applyHover();
          hideTip();
        })
        .transition()
        .duration(420)
        .ease(d3.easeCubicOut)
        .attr("y", (d) => bY(d.val))
        .attr("height", (d) => bH - bY(d.val));

      // value labels on bars
      const labels = allG.selectAll(".res-bar-label").data((d) => {
        const row = overall.find((r) => r.model_type === d);
        return row ? [{ model: d, val: row[activeMetric.key] }] : [];
      });
      labels
        .enter()
        .append("text")
        .attr("class", "res-bar-label")
        .attr("x", bX.bandwidth() / 2)
        .attr("text-anchor", "middle")
        .attr("font-size", "13px")
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("fill", (d) => colorOf(d.model))
        .merge(labels)
        .transition()
        .duration(420)
        .ease(d3.easeCubicOut)
        .attr("y", (d) => bY(d.val) - 5)
        .text((d) => d.val.toFixed(3));

      bars.exit().remove();
      labels.exit().remove();
      applyHover();
    }
    updateBars();

    // ════════════════════════════════
    // PLOT 2 — RADAR
    // ════════════════════════════════
    const RAXES = [
      { key: "mean_roc_auc", label: "ROC-AUC", min: 0.83, max: 0.935 },
      { key: "mean_pr_auc", label: "PR-AUC", min: 0.17, max: 0.28 },
      { key: "f1", label: "F1", min: 0.0, max: 0.56 },
      { key: "recall", label: "Recall", min: 0.0, max: 0.6 },
      { key: "precision", label: "Precision", min: 0.0, max: 0.86 },
    ];
    const RN = RAXES.length,
      rW = 440,
      rH = 330,
      rR = 110,
      rcx = rW / 2,
      rcy = rH / 2 + 6;

    const rSvg = d3
      .select("#res-radar-chart")
      .attr("viewBox", `0 0 ${rW} ${rH}`)
      .attr("width", "100%");
    const rG = rSvg.append("g");

    [0.25, 0.5, 0.75, 1].forEach((t) => {
      const pts = RAXES.map((_, i) => {
        const a = (i / RN) * 2 * Math.PI - Math.PI / 2;
        return [rcx + rR * t * Math.cos(a), rcy + rR * t * Math.sin(a)];
      });
      rG.append("polygon")
        .attr("points", pts.map((p) => p.join(",")).join(" "))
        .attr("fill", "none")
        .attr("stroke", "#e5e5ea")
        .attr("stroke-width", 0.5);
    });

    RAXES.forEach((ax, i) => {
      const a = (i / RN) * 2 * Math.PI - Math.PI / 2;
      rG.append("line")
        .attr("x1", rcx)
        .attr("y1", rcy)
        .attr("x2", rcx + rR * Math.cos(a))
        .attr("y2", rcy + rR * Math.sin(a))
        .attr("stroke", "#e5e5ea")
        .attr("stroke-width", 0.5);
      rG.append("text")
        .attr("x", rcx + (rR + 30) * Math.cos(a))
        .attr("y", rcy + (rR + 30) * Math.sin(a))
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("fill", "#333")
        .attr("font-size", "15px")
        .attr("font-weight", "500")
        .attr("font-family", "-apple-system,BlinkMacSystemFont,sans-serif")
        .text(ax.label);
    });

    MODELS.forEach((m) => {
      const row = overall.find((r) => r.model_type === m.key);
      if (!row) return;
      const pts = RAXES.map((ax, i) => {
        const t = Math.max(
          0,
          Math.min(1, (row[ax.key] - ax.min) / (ax.max - ax.min)),
        );
        const a = (i / RN) * 2 * Math.PI - Math.PI / 2;
        return [rcx + rR * t * Math.cos(a), rcy + rR * t * Math.sin(a)];
      });
      rG.append("polygon")
        .datum({ key: m.key })
        .attr("class", "res-radar-poly")
        .attr("points", pts.map((p) => p.join(",")).join(" "))
        .attr("fill", m.color)
        .attr("fill-opacity", 0.1)
        .attr("stroke", m.color)
        .attr("stroke-width", 2)
        .style("cursor", "pointer")
        .on("mouseover", (ev) => {
          hovered = m.key;
          applyHover();
          showTip(makeTip(m.key, row), ev);
        })
        .on("mousemove", moveTip)
        .on("mouseout", () => {
          hovered = null;
          applyHover();
          hideTip();
        });
    });

    // ════════════════════════════════
    // PLOT 3 — HEATMAP
    // ════════════════════════════════
    // Larger cells so numbers are readable
    const cellW = 54,
      cellH = 38;
    const hmM = { top: 24, right: 24, bottom: 120, left: 80 };
    const hmW = TARGETS.length * cellW,
      hmH = MODEL_KEYS.length * cellH;

    const hmSvg = d3
      .select("#res-heatmap")
      .attr(
        "viewBox",
        `0 0 ${hmW + hmM.left + hmM.right} ${hmH + hmM.top + hmM.bottom}`,
      )
      .attr("width", Math.max(900, hmW + hmM.left + hmM.right))
      .style("min-width", "900px");

    const hmG = hmSvg
      .append("g")
      .attr("transform", `translate(${hmM.left},${hmM.top})`);

    // High-contrast diverging color: white at midpoint, blue at top
    // Use a 3-stop scale: low=warm amber, mid=white, high=blue
    const hmColor = d3
      .scaleLinear()
      .domain([0.74, 0.855, 0.96])
      .range(["#fef3c7", "#ffffff", "#3b82f6"])
      .clamp(true);

    const cellData = [];
    MODEL_KEYS.forEach((mk) =>
      TARGETS.forEach((t) => {
        const row = perTarget.find(
          (r) => r.model_type === mk && r.target === t,
        );
        cellData.push({ model: mk, target: t, row: row || null });
      }),
    );

    // cells
    hmG
      .selectAll(".res-hm-cell")
      .data(cellData)
      .enter()
      .append("rect")
      .attr("class", "res-hm-cell")
      .attr("x", (d) => TARGETS.indexOf(d.target) * cellW + 1)
      .attr("y", (d) => MODEL_KEYS.indexOf(d.model) * cellH + 1)
      .attr("width", cellW - 2)
      .attr("height", cellH - 2)
      .attr("rx", 3)
      .attr("fill", (d) => (d.row ? hmColor(d.row.roc_auc) : "#f5f5f7"))
      .style("cursor", "pointer")
      .on("mouseover", (ev, d) => {
        if (!d.row) return;
        hovered = d.model;
        applyHover();
        showTip(
          makeTip(d.model, d.row, TARGET_LABELS[d.target] || d.target),
          ev,
        );
      })
      .on("mousemove", moveTip)
      .on("mouseout", () => {
        hovered = null;
        applyHover();
        hideTip();
      })
      .on("click", (_, d) => {
        if (d.row) showDetailCell(d);
      });

    // value text — dark text on light cells, white on dark cells
    hmG
      .selectAll(".res-hm-val")
      .data(cellData)
      .enter()
      .append("text")
      .attr("class", "res-hm-val")
      .attr("x", (d) => TARGETS.indexOf(d.target) * cellW + cellW / 2)
      .attr("y", (d) => MODEL_KEYS.indexOf(d.model) * cellH + cellH / 2 + 1)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", "10px")
      .attr("font-weight", "600")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("fill", (d) => {
        if (!d.row) return "#aaa";
        // dark text for light cells (mid range), white for deep blue cells
        return d.row.roc_auc > 0.91 ? "#fff" : "#1d1d1f";
      })
      .attr("pointer-events", "none")
      .text((d) => (d.row ? d.row.roc_auc.toFixed(2) : "—"));

    // row labels (model short names)
    MODEL_KEYS.forEach((mk, i) => {
      const m = modelOf(mk);
      hmG
        .append("text")
        .attr("x", -8)
        .attr("y", i * cellH + cellH / 2 + 1)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "11px")
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-weight", "500")
        .attr("fill", m.color)
        .style("cursor", "pointer")
        .text(m.short)
        .on("mouseover", () => {
          hovered = mk;
          applyHover();
        })
        .on("mouseout", () => {
          hovered = null;
          applyHover();
        });
    });

    // column labels (diagnosis names) — angled
    TARGETS.forEach((t, i) => {
      hmG
        .append("text")
        .attr(
          "transform",
          `translate(${i * cellW + cellW / 2},${hmH + 10}) rotate(40)`,
        )
        .attr("text-anchor", "start")
        .attr("font-size", "11px")
        .attr("font-family", "-apple-system,sans-serif")
        .attr("fill", "#444")
        .style("cursor", "pointer")
        .text(TARGET_LABELS[t] || t)
        .on("click", () => showDetailColumn(t));
    });

    // colour scale legend bar
    const defs = hmSvg.append("defs");
    const gid = "hmGradRes4";
    const lg = defs.append("linearGradient").attr("id", gid);
    lg.append("stop").attr("offset", "0%").attr("stop-color", "#fef3c7");
    lg.append("stop").attr("offset", "50%").attr("stop-color", "#ffffff");
    lg.append("stop").attr("offset", "100%").attr("stop-color", "#3b82f6");
    const lgG = hmSvg
      .append("g")
      .attr("transform", `translate(${hmM.left},${hmM.top + hmH + 96})`);
    lgG
      .append("rect")
      .attr("width", 130)
      .attr("height", 8)
      .attr("rx", 4)
      .attr("fill", `url(#${gid})`)
      .attr("stroke", "#e5e5ea")
      .attr("stroke-width", 0.5);
    lgG
      .append("text")
      .attr("x", 0)
      .attr("y", -6)
      .attr("font-size", "10px")
      .attr("font-family", "JetBrains Mono,monospace")
      .attr("fill", "#86868b")
      .text("0.74");
    lgG
      .append("text")
      .attr("x", 65)
      .attr("y", -6)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("font-family", "JetBrains Mono,monospace")
      .attr("fill", "#86868b")
      .text("0.855");
    lgG
      .append("text")
      .attr("x", 130)
      .attr("y", -6)
      .attr("text-anchor", "end")
      .attr("font-size", "10px")
      .attr("font-family", "JetBrains Mono,monospace")
      .attr("fill", "#86868b")
      .text("0.96 →");
    lgG
      .append("text")
      .attr("x", 0)
      .attr("y", 20)
      .attr("font-size", "10px")
      .attr("font-family", "-apple-system,sans-serif")
      .attr("fill", "#86868b")
      .text("ROC-AUC");

    // ── DETAIL PANEL ────────────────────────────────
    const detailEl = document.getElementById("res-detail");
    const DBARS = [
      { key: "roc_auc", label: "ROC-AUC", color: "#4f8ef7", max: 1 },
      { key: "pr_auc", label: "PR-AUC", color: "#a78bfa", max: 0.85 },
      { key: "f1", label: "F1", color: "#34d399", max: 1 },
      { key: "precision", label: "Precision", color: "#fb923c", max: 1 },
      { key: "recall", label: "Recall", color: "#f472b6", max: 1 },
    ];

    function showDetailCell(d) {
      const r = d.row,
        m = modelOf(d.model);
      detailEl.innerHTML = `
        <div class="res-detail-title">
          <span style="color:${m.color}">${m.label}</span>
          <span style="color:#86868b;font-weight:400"> · ${TARGET_LABELS[d.target] || d.target}</span>
        </div>
        <div class="res-detail-bars">
          ${DBARS.map((db) => {
            const val = r[db.key] ?? 0,
              pct = Math.min(100, (val / db.max) * 100);
            return `<div class="res-dbar-row">
              <div class="res-dbar-label">${db.label}</div>
              <div class="res-dbar-track"><div class="res-dbar-fill" style="width:${pct}%;background:${db.color}"></div></div>
              <div class="res-dbar-val">${val.toFixed(3)}</div>
            </div>`;
          }).join("")}
        </div>`;
    }

    function showDetailColumn(target) {
      const label = TARGET_LABELS[target] || target;
      const rows = MODEL_KEYS.map((mk) => ({
        model: mk,
        r: perTarget.find((x) => x.model_type === mk && x.target === target),
      }))
        .filter((x) => x.r)
        .sort((a, b) => b.r.roc_auc - a.r.roc_auc);
      detailEl.innerHTML = `
        <div class="res-detail-title">
          ${label}<span style="color:#86868b;font-weight:400"> · ROC-AUC ranking</span>
        </div>
        <div class="res-detail-bars">
          ${rows
            .map((x, i) => {
              const m = modelOf(x.model);
              const pct = Math.min(
                100,
                ((x.r.roc_auc - 0.74) / (0.96 - 0.74)) * 100,
              );
              return `<div class="res-dbar-row">
              <div class="res-dbar-label" style="color:${m.color}">#${i + 1} ${m.short}</div>
              <div class="res-dbar-track"><div class="res-dbar-fill" style="width:${pct}%;background:${m.color}"></div></div>
              <div class="res-dbar-val">${x.r.roc_auc.toFixed(3)}</div>
            </div>`;
            })
            .join("")}
        </div>`;
    }
  } // end init()
})();
