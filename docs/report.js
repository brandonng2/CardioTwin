/* D3.js vital signs background + scroll animations + stat counters */

document.addEventListener("DOMContentLoaded", () => {
  initVitalsBackground();
  initScrollAnimations();
  initStatCounters();
});

/* D3.js ECG-style vital signs waveform background */
function initVitalsBackground() {
  const container = document.getElementById("vitals-bg");
  if (!container || typeof d3 === "undefined") return;

  const hero = container.closest(".hero");
  const width = hero ? hero.offsetWidth : window.innerWidth;
  const height = hero ? hero.offsetHeight : 500;
  const numLines = 4;
  const ptsPerBeat = 120;
  const beatWidth = width * 0.18;
  const totalWidth = width * 2;
  const numBeats = Math.ceil(totalWidth / beatWidth) + 2;

  // Physiologically accurate ECG waveforms — different morphologies per lead
  // t: 0–1 over one cardiac cycle (~0.8s at 72 bpm)

  function leadII(t) {
    const b = t % 1;
    if (b < 0.06) return 0;
    if (b < 0.12) return 0.22 * Math.sin((b - 0.06) * 52.4); // P wave
    if (b < 0.18) return 0; // PR segment
    if (b < 0.2) return -0.15 * Math.exp(-Math.pow((b - 0.185) * 80, 2)); // Q
    if (b < 0.26) return 1.0 * Math.exp(-Math.pow((b - 0.23) * 55, 2)); // R spike
    if (b < 0.28) return -0.2 * Math.exp(-Math.pow((b - 0.27) * 70, 2)); // S
    if (b < 0.34) return 0; // ST segment
    if (b < 0.5) return 0.28 * Math.sin((b - 0.34) * 28.6); // T wave
    return 0;
  }

  function leadAVR(t) {
    const b = t % 1;
    if (b < 0.06) return 0;
    if (b < 0.12) return -0.2 * Math.sin((b - 0.06) * 52.4); // P inverted
    if (b < 0.18) return 0;
    if (b < 0.22) return 0.4 * Math.exp(-Math.pow((b - 0.2) * 50, 2)); // dominant S in aVR
    if (b < 0.3) return -0.35 * Math.exp(-Math.pow((b - 0.26) * 45, 2)); // deep Q
    if (b < 0.34) return 0;
    if (b < 0.5) return -0.22 * Math.sin((b - 0.34) * 28.6); // T inverted
    return 0;
  }

  function leadV1(t) {
    const b = t % 1;
    if (b < 0.06) return 0;
    if (b < 0.12) return 0.12 * Math.sin((b - 0.06) * 52.4); // small P
    if (b < 0.18) return 0;
    if (b < 0.21) return 0.35 * Math.exp(-Math.pow((b - 0.195) * 65, 2)); // small r
    if (b < 0.3) return -0.7 * Math.exp(-Math.pow((b - 0.265) * 40, 2)); // deep S (rS)
    if (b < 0.36) return 0;
    if (b < 0.5) return 0.15 * Math.sin((b - 0.36) * 25); // small T
    return 0;
  }

  function leadV5(t) {
    const b = t % 1;
    if (b < 0.06) return 0;
    if (b < 0.12) return 0.18 * Math.sin((b - 0.06) * 52.4); // P
    if (b < 0.18) return 0;
    if (b < 0.195) return -0.12 * Math.exp(-Math.pow((b - 0.188) * 90, 2)); // q
    if (b < 0.26) return 0.95 * Math.exp(-Math.pow((b - 0.225) * 50, 2)); // tall R (qR)
    if (b < 0.3) return -0.08 * Math.exp(-Math.pow((b - 0.285) * 60, 2)); // small s
    if (b < 0.35) return 0;
    if (b < 0.5) return 0.32 * Math.sin((b - 0.35) * 26); // prominent T
    return 0;
  }

  const waveformFns = [leadII, leadAVR, leadV1, leadV5];

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "none");

  const defs = svg.append("defs");
  const filter = defs
    .append("filter")
    .attr("id", "ecg-glow")
    .attr("x", "-20%")
    .attr("y", "-20%")
    .attr("width", "140%")
    .attr("height", "140%");
  filter
    .append("feGaussianBlur")
    .attr("stdDeviation", 1)
    .attr("result", "blur");
  filter
    .append("feMerge")
    .selectAll("feMergeNode")
    .data(["blur", "SourceGraphic"])
    .enter()
    .append("feMergeNode")
    .attr("in", (d) => d);

  const gridG = svg.append("g").attr("class", "ecg-grid");
  const gridSize = 40;
  for (let x = 0; x <= width + gridSize; x += gridSize) {
    gridG
      .append("line")
      .attr("x1", x)
      .attr("y1", 0)
      .attr("x2", x)
      .attr("y2", height)
      .attr("stroke", "rgba(0, 255, 136, 0.08)")
      .attr("stroke-width", 1);
  }
  for (let y = 0; y <= height + gridSize; y += gridSize) {
    gridG
      .append("line")
      .attr("x1", 0)
      .attr("y1", y)
      .attr("x2", width)
      .attr("y2", y)
      .attr("stroke", "rgba(0, 255, 136, 0.08)")
      .attr("stroke-width", 1);
  }

  const g = svg.append("g");
  const lines = [];
  const lineColors = ["#00ff88", "#00e676", "#69f0ae", "#1de9b6"];
  for (let i = 0; i < numLines; i++) {
    lines.push(
      g
        .append("path")
        .attr("fill", "none")
        .attr("stroke", lineColors[i])
        .attr("stroke-width", 4)
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .attr("filter", "url(#ecg-glow)"),
    );
  }

  const lineGen = d3
    .line()
    .x((d) => d.x)
    .y((d) => d.y)
    .curve(d3.curveMonotoneX);

  function update() {
    const now = Date.now() / 1000;
    const scroll = (now * 0.15 * beatWidth) % beatWidth;

    for (let lineIdx = 0; lineIdx < numLines; lineIdx++) {
      const points = [];
      const lineSpacing = height / (numLines + 1);
      const centerY = lineSpacing * (lineIdx + 1);
      const amp = lineSpacing * 0.38;
      const phase = (lineIdx * 0.15) % 1;

      const ecgFn = waveformFns[lineIdx];
      for (let b = -1; b < numBeats; b++) {
        for (let p = 0; p < ptsPerBeat; p++) {
          const beatT = (p / ptsPerBeat + phase) % 1;
          const x = b * beatWidth + (p / ptsPerBeat) * beatWidth - scroll;
          if (x >= -20 && x <= width + 20) {
            points.push({ x, y: centerY + ecgFn(beatT) * amp });
          }
        }
      }
      points.sort((a, b) => a.x - b.x);
      lines[lineIdx].attr("d", lineGen(points));
    }
  }

  update();
  d3.timer(update);
}

function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { rootMargin: "0px 0px -80px 0px", threshold: 0.1 },
  );

  document
    .querySelectorAll(".animate-on-scroll")
    .forEach((el) => observer.observe(el));
}

function initStatCounters() {
  const statValues = document.querySelectorAll(".stat-value[data-target]");

  const countUp = (el, target) => {
    const duration = 1500;
    const start = 0;
    const startTime = performance.now();

    const update = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = Math.floor(start + (target - start) * eased);
      el.textContent = value;
      if (progress < 1) requestAnimationFrame(update);
    };

    requestAnimationFrame(update);
  };

  const statObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const target = parseInt(el.dataset.target, 10);
          if (!el.dataset.animated) {
            countUp(el, target);
            el.dataset.animated = "true";
          }
        }
      });
    },
    { threshold: 0.5 },
  );

  statValues.forEach((el) => statObserver.observe(el));
}
