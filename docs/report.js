/* D3.js vital signs background + scroll animations + stat counters */
/* Version 2.0 - Updated to use external CSV file */
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";

document.addEventListener("DOMContentLoaded", () => {
  initVitalsBackground();
  initScrollAnimations();
  initCVDCounter();
  initDeathsChart();
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
    // P wave: 0.08-0.12s duration, rounded
    if (b >= 0.08 && b < 0.2) {
      const pPhase = (b - 0.08) / 0.12;
      return 0.25 * Math.sin(pPhase * Math.PI);
    }
    // PR segment: isoelectric
    if (b >= 0.2 && b < 0.24) return 0;
    // Q wave: small negative deflection
    if (b >= 0.24 && b < 0.26) {
      const qPhase = (b - 0.24) / 0.02;
      return -0.15 * Math.sin(qPhase * Math.PI);
    }
    // R wave: tall sharp positive deflection
    if (b >= 0.26 && b < 0.3) {
      const rPhase = (b - 0.26) / 0.04;
      return 1.2 * Math.exp(-Math.pow((rPhase - 0.5) * 8, 2));
    }
    // S wave: small negative deflection
    if (b >= 0.3 && b < 0.33) {
      const sPhase = (b - 0.3) / 0.03;
      return -0.2 * Math.sin(sPhase * Math.PI);
    }
    // ST segment: slightly elevated isoelectric
    if (b >= 0.33 && b < 0.38) return 0.02;
    // T wave: rounded positive deflection
    if (b >= 0.38 && b < 0.54) {
      const tPhase = (b - 0.38) / 0.16;
      return 0.35 * Math.sin(tPhase * Math.PI);
    }
    // TP segment: isoelectric baseline
    return 0;
  }

  function leadAVR(t) {
    const b = t % 1;
    // P wave: inverted in aVR
    if (b >= 0.08 && b < 0.2) {
      const pPhase = (b - 0.08) / 0.12;
      return -0.18 * Math.sin(pPhase * Math.PI);
    }
    if (b >= 0.2 && b < 0.24) return 0;
    // QRS: predominantly negative in aVR
    if (b >= 0.24 && b < 0.26) {
      const qPhase = (b - 0.24) / 0.02;
      return 0.25 * Math.sin(qPhase * Math.PI);
    }
    if (b >= 0.26 && b < 0.3) {
      const rPhase = (b - 0.26) / 0.04;
      return -0.9 * Math.exp(-Math.pow((rPhase - 0.5) * 8, 2));
    }
    if (b >= 0.3 && b < 0.33) {
      const sPhase = (b - 0.3) / 0.03;
      return 0.15 * Math.sin(sPhase * Math.PI);
    }
    if (b >= 0.33 && b < 0.38) return -0.02;
    // T wave: inverted in aVR
    if (b >= 0.38 && b < 0.54) {
      const tPhase = (b - 0.38) / 0.16;
      return -0.25 * Math.sin(tPhase * Math.PI);
    }
    return 0;
  }

  function leadV1(t) {
    const b = t % 1;
    // P wave: small upright
    if (b >= 0.08 && b < 0.2) {
      const pPhase = (b - 0.08) / 0.12;
      return 0.15 * Math.sin(pPhase * Math.PI);
    }
    if (b >= 0.2 && b < 0.24) return 0;
    // rS pattern: small r, deep S
    if (b >= 0.24 && b < 0.26) {
      const rPhase = (b - 0.24) / 0.02;
      return 0.3 * Math.sin(rPhase * Math.PI);
    }
    if (b >= 0.26 && b < 0.32) {
      const sPhase = (b - 0.26) / 0.06;
      return -0.85 * Math.exp(-Math.pow((sPhase - 0.5) * 6, 2));
    }
    if (b >= 0.32 && b < 0.38) return 0;
    // T wave: upright but small
    if (b >= 0.38 && b < 0.54) {
      const tPhase = (b - 0.38) / 0.16;
      return 0.2 * Math.sin(tPhase * Math.PI);
    }
    return 0;
  }

  function leadV5(t) {
    const b = t % 1;
    // P wave: upright
    if (b >= 0.08 && b < 0.2) {
      const pPhase = (b - 0.08) / 0.12;
      return 0.22 * Math.sin(pPhase * Math.PI);
    }
    if (b >= 0.2 && b < 0.24) return 0;
    // qR pattern: small q, tall R
    if (b >= 0.24 && b < 0.26) {
      const qPhase = (b - 0.24) / 0.02;
      return -0.1 * Math.sin(qPhase * Math.PI);
    }
    if (b >= 0.26 && b < 0.32) {
      const rPhase = (b - 0.26) / 0.06;
      return 1.1 * Math.exp(-Math.pow((rPhase - 0.5) * 6, 2));
    }
    if (b >= 0.32 && b < 0.34) {
      const sPhase = (b - 0.32) / 0.02;
      return -0.08 * Math.sin(sPhase * Math.PI);
    }
    if (b >= 0.34 && b < 0.38) return 0.02;
    // T wave: prominent upright
    if (b >= 0.38 && b < 0.54) {
      const tPhase = (b - 0.38) / 0.16;
      return 0.4 * Math.sin(tPhase * Math.PI);
    }
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

  const gridG = svg.append("g").attr("class", "ecg-grid");
  const gridSize = 50;
  for (let x = 0; x <= width + gridSize; x += gridSize) {
    gridG
      .append("line")
      .attr("x1", x)
      .attr("y1", 0)
      .attr("x2", x)
      .attr("y2", height)
      .attr("stroke", "rgba(255, 255, 255, 0.05)")
      .attr("stroke-width", 0.5);
  }
  for (let y = 0; y <= height + gridSize; y += gridSize) {
    gridG
      .append("line")
      .attr("x1", 0)
      .attr("y1", y)
      .attr("x2", width)
      .attr("y2", y)
      .attr("stroke", "rgba(255, 255, 255, 0.05)")
      .attr("stroke-width", 0.5);
  }

  const g = svg.append("g");
  const lines = [];
  const lineColors = ["#ff6482", "#af52de", "#bf5af2", "#ff8fa3"];
  for (let i = 0; i < numLines; i++) {
    lines.push(
      g
        .append("path")
        .attr("fill", "none")
        .attr("stroke", lineColors[i])
        .attr("stroke-width", 2)
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .style("opacity", 0.7),
    );
  }

  const lineGen = d3
    .line()
    .x((d) => d.x)
    .y((d) => d.y)
    .curve(d3.curveLinear);

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
        } else {
          entry.target.classList.remove("visible");
        }
      });
    },
    { rootMargin: "0px 0px -80px 0px", threshold: 0.1 },
  );

  document
    .querySelectorAll(".animate-on-scroll")
    .forEach((el) => observer.observe(el));
}

function initCVDCounter() {
  const cvdCounter = document.getElementById("cvd-counter");
  if (!cvdCounter) return;

  const target = 20.5;
  const duration = 2500;
  let animationFrame = null;

  const animate = () => {
    const startTime = performance.now();

    const update = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = (target * eased).toFixed(1);
      cvdCounter.textContent = value + "M";

      if (progress < 1) {
        animationFrame = requestAnimationFrame(update);
      }
    };

    animationFrame = requestAnimationFrame(update);
  };

  const reset = () => {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    cvdCounter.textContent = "0";
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animate();
        } else {
          reset();
        }
      });
    },
    { threshold: 0.3 },
  );

  observer.observe(cvdCounter);
}

function initDeathsChart() {
  const container = document.getElementById("deaths-chart");
  if (!container || typeof d3 === "undefined") return;

  // Load CSV data from file
  d3.csv("top_10_causes_of_death.csv").then((data) => {
    renderDeathsChart(data, container);
  }).catch((error) => {
    console.error("Error loading CSV:", error);
  });
}

function renderDeathsChart(data, container) {
  const grouped = d3.group(data, (d) => d.cause);

  // Filter to keep only the top 5 causes by total deaths in 2023
  const causeTotals = Array.from(grouped.entries()).map(([cause, values]) => {
    const latest = values.find(d => d.year === "2023");
    return {
      cause,
      total: latest ? +latest.deaths : 0
    };
  });
  
  // Sort and keep top 5
  const topCauses = causeTotals
    .sort((a, b) => b.total - a.total)
    .slice(0, 5)
    .map(d => d.cause);
  
  // Filter grouped data to only include top causes
  const filteredGrouped = new Map(
    Array.from(grouped.entries()).filter(([cause]) => topCauses.includes(cause))
  );

  // Create a color gradient from ECG primary to secondary
  const primaryColor = "#ff6482";  // ecg-primary
  const secondaryColor = "#af52de"; // ecg-secondary (darker)
  
  // Generate colors interpolating between primary and secondary
  const colorScale = d3.interpolateRgb(primaryColor, secondaryColor);
  const causes = Array.from(filteredGrouped.keys());
  const colorMap = {};
  
  causes.forEach((cause, i) => {
    const t = i / Math.max(1, causes.length - 1); // 0 to 1
    colorMap[cause] = colorScale(t);
  });
  
  // Ensure Cardiovascular diseases gets the primary color
  if (colorMap["Cardiovascular diseases"]) {
    colorMap["Cardiovascular diseases"] = primaryColor;
  }

  const margin = { top: 30, right: 30, bottom: 90, left: 70 };
  let svg = null;
  let chartElements = null;

  const createChart = () => {
    // Clear any existing chart
    d3.select(container).selectAll("*").remove();

    const containerWidth = container.offsetWidth;
    const containerHeight = 450;
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;

    svg = d3
      .select(container)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${containerWidth} ${containerHeight}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear().domain([1980, 2023]).range([0, width]);

    const y = d3.scaleLinear().domain([0, 20000000]).range([height, 0]);

    // X Axis
    svg
      .append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(8))
      .style("color", "#86868b")
      .style("font-size", "12px");

    // X Axis Label
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height + 40)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "#86868b")
      .style("font-weight", "600")
      .text("Year");

    // Y Axis
    svg
      .append("g")
      .call(
        d3
          .axisLeft(y)
          .tickFormat((d) => `${d / 1000000}M`)
          .ticks(5),
      )
      .style("color", "#86868b")
      .style("font-size", "12px");

    // Y Axis Label
    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -50)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("fill", "#86868b")
      .style("font-weight", "600")
      .text("Deaths (millions)");

    // Line generator
    const line = d3
      .line()
      .x((d) => x(+d.year))
      .y((d) => y(+d.deaths))
      .curve(d3.curveMonotoneX);

    // Create tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "chart-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background", "rgba(255, 255, 255, 0.95)")
      .style("padding", "12px 16px")
      .style("border-radius", "8px")
      .style("box-shadow", "0 4px 12px rgba(0,0,0,0.15)")
      .style("font-size", "13px")
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("border", "1px solid #d2d2d7");

    // Draw lines for each cause with animation
    filteredGrouped.forEach((values, cause) => {
      const sortedValues = values.sort((a, b) => +a.year - +b.year);

      const path = svg
        .append("path")
        .datum(sortedValues)
        .attr("fill", "none")
        .attr("stroke", colorMap[cause] || "#d2d2d7")
        .attr("stroke-width", cause === "Cardiovascular diseases" ? 3 : 2.5)
        .attr("d", line);

      // Animate line drawing
      const totalLength = path.node().getTotalLength();

      path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .style("opacity", 0.85)
        .transition()
        .duration(1500)
        .ease(d3.easeLinear)
        .attr("stroke-dashoffset", 0);

      // Add invisible wider line for easier hovering
      const hoverPath = svg
        .append("path")
        .datum(sortedValues)
        .attr("fill", "none")
        .attr("stroke", "transparent")
        .attr("stroke-width", 20)
        .attr("d", line)
        .style("cursor", "pointer")
        .on("mouseover", function(event) {
          // Highlight the line
          path.style("opacity", 1).attr("stroke-width", cause === "Cardiovascular diseases" ? 4 : 3.5);
          tooltip.style("visibility", "visible");
        })
        .on("mousemove", function(event) {
          // Find closest data point
          const [mouseX] = d3.pointer(event, svg.node());
          const year = Math.round(x.invert(mouseX));
          const dataPoint = sortedValues.find(d => +d.year === year) || sortedValues[0];
          
          const deaths = (+dataPoint.deaths / 1000000).toFixed(2);
          
          tooltip
            .html(`
              <div style="font-weight: 600; color: ${colorMap[cause]}; margin-bottom: 6px;">${cause}</div>
              <div style="color: #1d1d1f;"><strong>Year:</strong> ${dataPoint.year}</div>
              <div style="color: #1d1d1f;"><strong>Deaths:</strong> ${deaths}M</div>
            `)
            .style("top", (event.pageY - 10) + "px")
            .style("left", (event.pageX + 10) + "px");
        })
        .on("mouseout", function() {
          path.style("opacity", 0.85).attr("stroke-width", cause === "Cardiovascular diseases" ? 3 : 2.5);
          tooltip.style("visibility", "hidden");
        });
    });

    // Add horizontal legend below the chart with 2-2-1 layout
    const legend = svg
      .append("g")
      .attr("transform", `translate(0, ${height + 55})`);

    Object.entries(colorMap).forEach(([cause, color], i) => {
      let xPos, yPos;
      
      if (i < 2) {
        // First row (2 items) - -30 and 180
        xPos = i === 0 ? -30 : 180;
        yPos = 0;
      } else if (i < 4) {
        // Second row (2 items) - -30 and 180
        const secondRowIndex = i - 2;
        xPos = secondRowIndex === 0 ? -30 : 180;
        yPos = 24;
      } else {
        // Third row (1 item) - -30
        xPos = -30;
        yPos = 48;
      }
      
      const g = legend
        .append("g")
        .attr("transform", `translate(${xPos}, ${yPos})`)
        .style("opacity", 0);

      g.append("line")
        .attr("x1", 0)
        .attr("x2", 24)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", color)
        .attr("stroke-width", cause === "Cardiovascular diseases" ? 3 : 2.5);

      g.append("text")
        .attr("x", 30)
        .attr("y", 4)
        .text(cause)
        .style("font-size", "12px")
        .style("fill", "#1d1d1f")
        .style("font-weight", cause === "Cardiovascular diseases" ? "600" : "400");

      // Fade in legend items
      g.transition()
        .delay(500 + i * 100)
        .duration(500)
        .style("opacity", 1);
    });
  };

  const clearChart = () => {
    d3.select(container).selectAll("*").remove();
  };

  let isChartCreated = false;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !isChartCreated) {
          isChartCreated = true;
          createChart();
        }
      });
    },
    { threshold: 0.2 },
  );

  observer.observe(container);
}