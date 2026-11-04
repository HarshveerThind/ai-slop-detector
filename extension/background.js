// All scan work happens here because the popup closes when the user clicks away.
// State lives in chrome.storage.session keyed by video ID, the popup only renders it.
// The worker can be shut down at any time, so nothing important is kept in variables.

importScripts("config.js");

const STALE_MS = { recording: 15000, scoring: 120000 };
const BADGE_COLORS = { likely_ai: "#d93025", unsure: "#80868b", likely_human: "#188038" };

function stateKey(videoId) {
  return `video:${videoId}`;
}

async function getState(videoId) {
  const key = stateKey(videoId);
  const stored = await chrome.storage.session.get(key);
  return stored[key] || { state: "idle" };
}

async function setState(videoId, state) {
  await chrome.storage.session.set({
    [stateKey(videoId)]: { ...state, updatedAt: Date.now() },
  });
}

async function setBadge(tabId, result) {
  if (tabId == null) return;
  try {
    if (!result) {
      await chrome.action.setBadgeText({ tabId, text: "" });
      return;
    }
    await chrome.action.setBadgeBackgroundColor({
      tabId,
      color: BADGE_COLORS[result.verdict] || BADGE_COLORS.unsure,
    });
    await chrome.action.setBadgeTextColor({ tabId, color: "#ffffff" });
    await chrome.action.setBadgeText({ tabId, text: `${Math.round(result.overall * 100)}%` });
  } catch (e) {
    // the tab was closed while we were working
  }
}

async function finish(videoId, tabId, result) {
  await setState(videoId, { state: "done", result });
  await setBadge(tabId, result);
}

function isBusy(current) {
  const limit = STALE_MS[current.state];
  // a stale state means the worker or the page died partway, so it is safe to restart
  return limit !== undefined && Date.now() - (current.updatedAt || 0) < limit;
}

async function startScan(videoId, tabId, force) {
  const current = await getState(videoId);
  if (isBusy(current)) return;
  if (current.state === "done" && !force) {
    await setBadge(tabId, current.result);
    return;
  }

  await setState(videoId, { state: "scoring" });
  await setBadge(tabId, null);

  if (!force) {
    let response;
    try {
      response = await fetch(`${API_BASE}/score/${videoId}`);
    } catch (e) {
      await setState(videoId, { state: "error", error: "Could not reach the server" });
      return;
    }
    if (response.ok) {
      await finish(videoId, tabId, await response.json());
      return;
    }
    if (response.status !== 404) {
      await setState(videoId, { state: "error", error: await errorMessage(response) });
      return;
    }
  }

  await setState(videoId, { state: "recording", seconds: 0 });
  try {
    await chrome.tabs.sendMessage(tabId, { type: "record", videoId });
  } catch (e) {
    // no content script in this tab, for example it was open before the extension loaded
    await scoreVideo(videoId, tabId, null);
  }
}

async function errorMessage(response) {
  if (response.status === 429) return "Too many scans, try again in a minute";
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (e) {
    // no JSON body, use the generic message below
  }
  return "The server could not score this video";
}

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mimeType });
}

async function scoreVideo(videoId, tabId, clip) {
  await setState(videoId, { state: "scoring" });
  const form = new FormData();
  form.append("video_id", videoId);
  if (clip) form.append("audio", base64ToBlob(clip.audioBase64, clip.mimeType), "clip.webm");

  let response;
  try {
    response = await fetch(`${API_BASE}/score`, { method: "POST", body: form });
  } catch (e) {
    await setState(videoId, { state: "error", error: "Could not reach the server" });
    return;
  }
  if (!response.ok) {
    await setState(videoId, { state: "error", error: await errorMessage(response) });
    return;
  }
  await finish(videoId, tabId, await response.json());
}

async function handleMessage(message, sender) {
  const { type, videoId } = message;
  const tabId = sender.tab ? sender.tab.id : message.tabId;

  switch (type) {
    // from the popup
    case "startScan":
      await startScan(videoId, tabId, Boolean(message.force));
      break;
    case "scanWithoutAudio":
      await scoreVideo(videoId, tabId, null);
      break;

    // from the content script
    case "recordingProgress":
      await setState(videoId, {
        state: "recording",
        seconds: message.seconds,
        waiting: message.waiting || null,
      });
      break;
    case "clipReady":
      await scoreVideo(videoId, tabId, message);
      break;
    case "recordingFailed":
      await scoreVideo(videoId, tabId, null);
      break;
    case "recordingPaused":
      await setState(videoId, { state: "idle", notice: "paused" });
      break;
    case "recordingCancelled":
      await setState(videoId, { state: "idle" });
      break;
    case "navigated": {
      const current = videoId ? await getState(videoId) : null;
      await setBadge(tabId, current && current.state === "done" ? current.result : null);
      break;
    }
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender)
    .catch((e) => console.error("scan failed", e))
    .finally(() => sendResponse({ ok: true }));
  return true; // keeps the channel open until the work above is finished
});
