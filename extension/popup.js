// Renders whatever state the background worker has stored, and nothing else.

const RECORD_SECONDS = 20;
const VERDICT_TEXT = {
  likely_ai: "likely AI",
  unsure: "unsure",
  likely_human: "likely human",
};

const el = (id) => document.getElementById(id);

let videoId = null;
let tabId = null;

function videoIdFromUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (e) {
    return null;
  }
  if (!/(^|\.)youtube\.com$/.test(url.hostname)) return null;
  if (url.pathname === "/watch") return url.searchParams.get("v");
  const shorts = url.pathname.match(/^\/shorts\/([A-Za-z0-9_-]{11})/);
  return shorts ? shorts[1] : null;
}

function send(type, extra = {}) {
  chrome.runtime.sendMessage({ type, videoId, tabId, ...extra }).catch(() => {});
}

function showMessage(text, { progress = null, button = null, onClick = null } = {}) {
  el("result").hidden = true;
  el("message").hidden = false;
  el("message-text").textContent = text;

  el("progress").hidden = progress === null;
  if (progress !== null) el("progress-fill").style.width = `${progress * 100}%`;

  el("message-button").hidden = !button;
  if (button) {
    el("message-button").textContent = button;
    el("message-button").onclick = onClick;
  }
}

function renderScore(id, value) {
  const row = el(id);
  const available = value !== null && value !== undefined;
  row.querySelector(".score-value").textContent = available
    ? `${Math.round(value * 100)}%`
    : "not available";
  row.querySelector(".fill").style.width = available ? `${value * 100}%` : "0";
}

function showResult(result) {
  el("message").hidden = true;
  el("result").hidden = false;

  el("overall").textContent = `${Math.round(result.overall * 100)}%`;
  el("overall").className = result.verdict;
  el("verdict").textContent = VERDICT_TEXT[result.verdict] || result.verdict;
  el("verdict").className = result.verdict;

  renderScore("score-voice", result.voice);
  renderScore("score-visuals", result.visuals);
  renderScore("score-metadata", result.metadata);

  const reasons = el("reasons");
  reasons.replaceChildren(
    ...result.reasons.map((reason) => {
      const item = document.createElement("li");
      item.textContent = reason;
      return item;
    })
  );
}

function recordingText(current) {
  if (current.waiting === "ad") return "Waiting for the ad to finish";
  if (current.waiting === "paused") return "Paused. Play the video to keep listening";
  return `Listening to the audio, ${current.seconds || 0} of ${RECORD_SECONDS} seconds`;
}

function render(current) {
  switch (current.state) {
    case "recording":
      showMessage(recordingText(current), {
        progress: (current.seconds || 0) / RECORD_SECONDS,
      });
      break;
    case "scoring":
      showMessage("Scoring this video");
      break;
    case "done":
      showResult(current.result);
      break;
    case "error":
      showMessage(current.error || "Something went wrong", {
        button: "Try again",
        onClick: () => send("startScan", { force: true }),
      });
      break;
    default:
      if (current.notice === "paused") {
        showMessage("Play the video to scan its audio", {
          button: "Scan without audio",
          onClick: () => send("scanWithoutAudio"),
        });
      } else {
        showMessage("Starting scan");
      }
  }
}

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  videoId = tab ? videoIdFromUrl(tab.url || "") : null;
  tabId = tab ? tab.id : null;

  if (!videoId) {
    showMessage("Open a YouTube video to scan it");
    return;
  }

  const key = `video:${videoId}`;
  const stored = await chrome.storage.session.get(key);
  render(stored[key] || { state: "idle" });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "session" && changes[key] && changes[key].newValue) {
      render(changes[key].newValue);
    }
  });

  el("rescan").onclick = () => send("startScan", { force: true });
  send("startScan");
}

init();
