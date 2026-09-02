const resultViews = [32, 64, 128, 256];
const resultMetrics = [
  { label: "PSNR ↑", direction: "max" },
  { label: "SSIM ↑", direction: "max" },
  { label: "LPIPS ↓", direction: "min" },
];

const resultData = {
  "unposed-uncalibrated": {
    description: "Camera poses and intrinsics are estimated by the model.",
    rows: [
      { name: "YoNoSplat", offline: true, values: [22.368, 0.736, 0.180, 22.253, 0.732, 0.183, 21.827, 0.720, 0.194, 20.749, 0.677, 0.226] },
      { name: "AnySplat", offline: true, values: [19.944, 0.617, 0.240, 19.864, 0.609, 0.249, 19.566, 0.596, 0.264, null, null, null] },
      { name: "OF³GS", values: [17.002, 0.451, 0.479, 16.519, 0.432, 0.505, 16.095, 0.414, 0.530, 15.609, 0.399, 0.550] },
      { name: "KV Cache", values: [21.705, 0.703, 0.202, 21.302, 0.680, 0.223, 20.784, 0.655, 0.247, 19.819, 0.606, 0.292] },
      { name: "GIR", values: [21.752, 0.703, 0.203, 21.382, 0.682, 0.222, 20.933, 0.660, 0.245, 19.900, 0.608, 0.291] },
      { name: "ReCoSplat", ours: true, values: [22.097, 0.716, 0.194, 21.774, 0.696, 0.213, 21.284, 0.672, 0.235, 20.220, 0.617, 0.281] },
    ],
  },
  "unposed-calibrated": {
    description: "Camera intrinsics are provided; camera poses are estimated by the model.",
    rows: [
      { name: "YoNoSplat", offline: true, values: [22.575, 0.748, 0.177, 22.514, 0.747, 0.179, 22.053, 0.735, 0.189, 20.953, 0.692, 0.221] },
      { name: "S3PO-GS", values: [15.598, 0.399, 0.541, 16.231, 0.431, 0.539, 16.862, 0.451, 0.529, 16.809, 0.446, 0.533] },
      { name: "OF³GS", values: [17.062, 0.456, 0.476, 16.558, 0.437, 0.503, 16.134, 0.418, 0.527, 15.677, 0.403, 0.547] },
      { name: "KV Cache", values: [21.932, 0.717, 0.198, 21.544, 0.697, 0.218, 20.993, 0.670, 0.242, 20.007, 0.622, 0.286] },
      { name: "GIR", values: [21.947, 0.716, 0.199, 21.600, 0.697, 0.218, 21.107, 0.672, 0.241, 20.085, 0.622, 0.286] },
      { name: "ReCoSplat", ours: true, values: [22.417, 0.734, 0.188, 22.068, 0.713, 0.206, 21.576, 0.690, 0.227, 20.430, 0.633, 0.275] },
    ],
  },
  "posed-calibrated": {
    description: "Camera poses and intrinsics are provided.",
    rows: [
      { name: "YoNoSplat", offline: true, values: [22.998, 0.781, 0.162, 22.978, 0.784, 0.161, 22.597, 0.779, 0.167, 21.549, 0.749, 0.190] },
      { name: "ZPressor", offline: true, values: [19.293, 0.624, 0.269, 18.568, 0.586, 0.297, 17.992, 0.557, 0.321, 17.539, 0.532, 0.342] },
      { name: "KV Cache", values: [22.392, 0.758, 0.177, 22.210, 0.754, 0.184, 21.731, 0.739, 0.199, 20.694, 0.699, 0.235] },
      { name: "GIR", values: [22.478, 0.760, 0.177, 22.264, 0.755, 0.185, 21.779, 0.739, 0.200, 20.743, 0.699, 0.235] },
      { name: "ReCoSplat", ours: true, values: [23.084, 0.780, 0.164, 23.086, 0.782, 0.167, 22.852, 0.777, 0.176, 22.003, 0.751, 0.202] },
    ],
  },
};

const videoViews = [32, 64, 128, 256];
const videoFiles = {
  unposed_uncalib: {
    32: ["2beaca318994c25409dcbb6d0bdd96c3620f2f18aec44ea3f20edd302f18ca78", "35872363e17af5d173b6a0b09fcf5de94627ad5dc5f8a9ad4c579f3e70b4797a"],
    64: ["ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101", "d9b6376623741313bf6da6bf4cdb9828be614a2ce9390ceb3f31cd535d661a75"],
    128: ["91afb9910b042f7185c2b8e4b6b24b5785dae4542617b3f8005b5492f6d123f7", "71b2dc8a2aa553da09b8b94b9f0d5e8abcca307def74d26301616ee238464d46"],
    256: ["ceb252f5d4419510655cf9ed7afbf3e8e688825f798d80414c7715dc8ace153a", "1d6a9ed47cce39fd1c4d18f776bcc97e507b81bde921ca596bd91b0b02b5e414"],
  },
  unposed_calib: {
    32: ["26fd23358fa11fff0fb3180ef0b65591b486e20dcf753ce4a7aae49a37e370c7", "ff592398657b3dfe94153332861985194a3e3c9d199c4a3a27a0ce4038e81ade"],
    64: ["b92b499c9bf92327d5f3a44c9db49bde3400dcb1cfec48d6488831e2b304d0bb", "2cbfe28643b6636f9c70813cae7625aa858a352109493ac70fb429ce94dd55b3"],
    128: ["1de58be515696102c364b767f296600ffff853d4145a60dd30ece9d935317654", "0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb2307855afdaac8be"],
    256: ["ec305787b70029b782c71c1bf296c3885c7c22619e5661bd40085533ddfee5e4", "6e11e7f4fea305c7c4658d2c1f8df29e6f299330860cf48ffbf1c5ff8b96c0a8"],
  },
  posed_calib: {
    32: ["1264931635e127fb905c8953cbc2deadd0c763e633af7fbd9405a61ca849710c", "2b65ba886efac7af6253fce68ae9284bf4d4db019e17c47f3e853361acbdb066"],
    64: ["d8de66037bd03dd0d39d54f9978bac3318d912e30e22c21e1ada82a98ed48c53", "4ae797d07b6d1644c9db6919c8cc8c0d28d72be45108ac7a3abf8dc21b599d83"],
    128: ["cc08c0bdc34ddd2867705d0b17a86ec2a9d7c7926ce99070ed1fdc66a812de07", "14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49"],
    256: ["c076929db6501cf7ebe386c70e6d77ea3af844a745e794f2ec17c981c465a69b", "917e9c8985d353b0ee4c281f11fa84eb8550814562670cbb82cd0ec9c1194fe7"],
  },
};

function metricRanks(rows) {
  return Array.from({ length: resultViews.length * resultMetrics.length }, (_, column) => {
    const metric = resultMetrics[column % resultMetrics.length];
    const values = rows
      .filter((row) => !row.offline && row.values[column] !== null)
      .map((row) => row.values[column]);
    const sorted = [...new Set(values)].sort((a, b) => metric.direction === "max" ? b - a : a - b);
    return { best: sorted[0], second: sorted[1] };
  });
}

function renderResultsTable(setting) {
  const target = document.querySelector("#results-table");
  const description = document.querySelector("#results-setting-description");
  const data = resultData[setting];
  const ranks = metricRanks(data.rows);
  const groupHeaders = resultViews.map((view) => `<th colspan="3" scope="colgroup">${view} views</th>`).join("");
  const metricHeaders = resultViews.flatMap(() => resultMetrics.map((metric) => `<th scope="col">${metric.label}</th>`)).join("");
  const rows = data.rows.map((row) => {
    const rowClass = [row.offline ? "offline" : "", row.ours ? "ours" : ""].filter(Boolean).join(" ");
    const cells = row.values.map((value, column) => {
      if (value === null) {
        return '<td><abbr title="Not evaluated at the method\'s training resolution">N/A†</abbr></td>';
      }
      let rankClass = "";
      if (!row.offline && value === ranks[column].best) {
        rankClass = "metric-best";
      } else if (!row.offline && value === ranks[column].second) {
        rankClass = "metric-second";
      }
      return `<td class="${rankClass}">${value.toFixed(3)}</td>`;
    }).join("");
    return `<tr class="${rowClass}"><td class="method-cell">${row.name}</td>${cells}</tr>`;
  }).join("");

  target.innerHTML = `
    <table class="results-table">
      <caption class="sr-only">Novel view synthesis results on DL3DV for ${setting.replaceAll("-", " ")}</caption>
      <thead>
        <tr><th class="method-cell" rowspan="2" scope="col">Method</th>${groupHeaders}</tr>
        <tr>${metricHeaders}</tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  description.textContent = data.description;
}

let videoObserver;

function renderVideos(setting) {
  const target = document.querySelector("#video-grid");
  if (videoObserver) {
    videoObserver.disconnect();
  }
  target.querySelectorAll("video").forEach((video) => video.pause());

  const cards = [];
  for (let scene = 0; scene < 2; scene += 1) {
    for (const view of videoViews) {
      const filename = videoFiles[setting][view][scene];
      const source = `./static/videos/results/${setting}/${view}v/${filename}.mp4`;
      cards.push(`
        <article class="video-card paused">
          <video muted loop playsinline preload="metadata" aria-label="Scene ${scene + 1}, ${view}-view novel-view rendering">
            <source src="${source}" type="video/mp4">
          </video>
          <span class="video-state" aria-hidden="true">▶</span>
          <div class="video-caption"><strong>${view} views</strong><span>Scene ${scene + 1}</span></div>
        </article>`);
    }
  }
  target.innerHTML = cards.join("");

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  videoObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      const card = video.closest(".video-card");
      if (entry.isIntersecting && !reduceMotion && video.dataset.userPaused !== "true") {
        video.play().then(() => card.classList.remove("paused")).catch(() => card.classList.add("paused"));
      } else {
        video.pause();
        card.classList.add("paused");
      }
    });
  }, { threshold: 0.35 });

  target.querySelectorAll("video").forEach((video) => {
    const card = video.closest(".video-card");
    video.addEventListener("click", () => {
      if (video.paused) {
        video.dataset.userPaused = "false";
        video.play().then(() => card.classList.remove("paused")).catch(() => {});
      } else {
        video.dataset.userPaused = "true";
        video.pause();
        card.classList.add("paused");
      }
    });
    videoObserver.observe(video);
  });
}

function configureTabList(selector, dataAttribute, onChange) {
  const tabs = [...document.querySelectorAll(`${selector} [${dataAttribute}]`)];
  tabs.forEach((tab, index) => {
    tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) {
        return;
      }
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + offset + tabs.length) % tabs.length];
      activate(next);
      next.focus();
    });
  });

  function activate(activeTab) {
    tabs.forEach((tab) => {
      const selected = tab === activeTab;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    onChange(activeTab.getAttribute(dataAttribute));
  }
}

function configureNavigation() {
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    links.classList.toggle("open", open);
    document.body.classList.toggle("nav-open", open);
  });
  links.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      toggle.setAttribute("aria-expanded", "false");
      links.classList.remove("open");
      document.body.classList.remove("nav-open");
    });
  });
}

function configureCitationCopy() {
  const button = document.querySelector("[data-copy-citation]");
  button.addEventListener("click", async () => {
    const citation = document.querySelector("#bibtex").textContent;
    try {
      await navigator.clipboard.writeText(citation);
    } catch (error) {
      const field = document.createElement("textarea");
      field.value = citation;
      document.body.append(field);
      field.select();
      document.execCommand("copy");
      field.remove();
    }
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = "Copy BibTeX";
    }, 1800);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  configureNavigation();
  renderResultsTable("unposed-uncalibrated");
  renderVideos("unposed_uncalib");
  configureTabList(".results-tabs", "data-result-setting", renderResultsTable);
  configureTabList(".video-tabs", "data-video-setting", renderVideos);
  configureCitationCopy();
});
