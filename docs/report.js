/* D3.js vital signs background + scroll animations + stat counters */

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

  // Parse CSV data inline
  const csvData = `year,cause,deaths,deaths_lower,deaths_upper
1980,Cardiovascular diseases,12116561.690830473,11028346.853724385,13048659.658636544
1985,Cardiovascular diseases,12508892.646832569,11475136.302037831,13399628.763302252
1990,Cardiovascular diseases,13397894.064827552,12324095.870838743,14301438.088748962
1995,Cardiovascular diseases,14252863.10673066,13064253.093649315,15149734.267043978
2000,Cardiovascular diseases,15127530.885299476,13879623.950844133,16111774.64372334
2005,Cardiovascular diseases,15786072.36889781,14556042.65296635,16720668.82324493
2010,Cardiovascular diseases,16264606.172654845,15042527.090062413,17219098.65093289
2015,Cardiovascular diseases,17208352.084747426,16001323.110538438,18112386.37971564
2020,Cardiovascular diseases,17658557.794214997,16115057.799830234,18749602.69506749
2023,Cardiovascular diseases,19139018.09368501,17355526.65696515,20403359.75555195
1980,Total cancers,4778620.553689183,4525408.852813329,5123498.452049806
1985,Total cancers,5243088.295219312,4985351.092824577,5579725.939122331
1990,Total cancers,5818373.7682823,5531829.350125331,6195429.569206173
1995,Total cancers,6570664.943881275,6214779.086894074,6953063.732348664
2000,Total cancers,7197354.8802422695,6827085.349044554,7531287.283883136
2005,Total cancers,7646574.055173976,7237699.4734898405,7987127.8031253275
2010,Total cancers,8103866.875827074,7603834.042959072,8395699.093856633
2015,Total cancers,8620278.446351437,8044840.465108759,8976676.55472392
2020,Total cancers,9369755.232305313,8750021.063785639,9787012.972000783
2023,Total cancers,10434120.432555158,9605088.887119193,11031333.441804592
1980,Lower respiratory infections,3178213.2278480977,2770073.8375791293,3715304.185026179
1985,Lower respiratory infections,3102678.439698333,2706624.726674132,3521553.770520356
1990,Lower respiratory infections,3068959.6374048907,2710883.8869682537,3475802.1311421823
1995,Lower respiratory infections,3180251.066062093,2750050.9766960074,3720127.3467134065
2000,Lower respiratory infections,2960683.868950551,2556663.318859992,3405870.5476991506
2005,Lower respiratory infections,2762654.5035363133,2401433.0950829215,3165458.7311464175
2010,Lower respiratory infections,2564326.7172251386,2271393.2885682466,2874621.348697646
2015,Lower respiratory infections,2550356.99843432,2282301.3591353777,2813512.75786936
2020,Lower respiratory infections,2214586.490139366,1971694.79492917,2512381.8461652542
2023,Lower respiratory infections,2501138.568273376,2240273.9879636248,2811711.568178583
1980,Neonatal disorders,3400099.9510319303,3161964.759499208,3639679.066678661
1985,Neonatal disorders,3335924.726877792,3166127.983560855,3538776.87990238
1990,Neonatal disorders,3244127.327889293,3069839.6863009022,3432903.9473033673
1995,Neonatal disorders,2948089.6324651763,2809659.8473064504,3094562.8621733887
2000,Neonatal disorders,2563621.5458113905,2450645.7024095766,2695089.4663324035
2005,Neonatal disorders,2279078.4969175486,2179899.3099316354,2390060.419072738
2010,Neonatal disorders,2157298.125267168,2065127.3354177482,2253394.1059172372
2015,Neonatal disorders,2018467.3728496935,1922053.3373190197,2117857.749870326
2020,Neonatal disorders,1763856.671129841,1655034.933439844,1863133.560294462
2023,Neonatal disorders,1627961.732945383,1506008.7124160386,1748006.3058008086`;

  const data = d3.csvParse(csvData);
  const grouped = d3.group(data, (d) => d.cause);

  const colorMap = {
    "Cardiovascular diseases": "#ff6482",
    "Total cancers": "#af52de",
    "Lower respiratory infections": "#bf5af2",
    "Neonatal disorders": "#ff8fa3",
  };

  const margin = { top: 30, right: 20, bottom: 50, left: 70 };
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
      .attr("width", containerWidth)
      .attr("height", containerHeight)
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

    // Draw lines for each cause with animation
    grouped.forEach((values, cause) => {
      const sortedValues = values.sort((a, b) => +a.year - +b.year);

      const path = svg
        .append("path")
        .datum(sortedValues)
        .attr("fill", "none")
        .attr("stroke", colorMap[cause] || "#d2d2d7")
        .attr("stroke-width", cause === "Cardiovascular diseases" ? 3 : 2)
        .attr("d", line);

      // Animate line drawing
      const totalLength = path.node().getTotalLength();

      path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .style("opacity", 0.9)
        .transition()
        .duration(1500)
        .ease(d3.easeLinear)
        .attr("stroke-dashoffset", 0);
    });

    // Add legend inside chart area
    const legend = svg
      .append("g")
      .attr("transform", `translate(${width - 200}, 10)`);

    Object.entries(colorMap).forEach(([cause, color], i) => {
      const g = legend
        .append("g")
        .attr("transform", `translate(0, ${i * 22})`)
        .style("opacity", 0);

      g.append("line")
        .attr("x1", 0)
        .attr("x2", 20)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", color)
        .attr("stroke-width", cause === "Cardiovascular diseases" ? 3 : 2);

      g.append("text")
        .attr("x", 26)
        .attr("y", 4)
        .text(cause)
        .style("font-size", "10px")
        .style("fill", "#1d1d1f")
        .style("font-weight", "400");

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

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          createChart();
        } else {
          clearChart();
        }
      });
    },
    { threshold: 0.2 },
  );

  observer.observe(container);
}