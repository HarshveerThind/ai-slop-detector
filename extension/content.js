// Records a short clip of the audio already playing in the page.
// YouTube is a single page app, so this script stays alive across videos.

const RECORD_SECONDS = 20;
const AD_WAIT_SECONDS = 60;
const MAX_HELD_SECONDS = 60; // paused or an ad partway through the recording
const MIME_TYPE = "audio/webm;codecs=opus";

let active = null; // the recording in progress, or a wait for an ad to end

function videoIdFromUrl() {
  const url = new URL(location.href);
  if (url.pathname === "/watch") return url.searchParams.get("v");
  const shorts = url.pathname.match(/^\/shorts\/([A-Za-z0-9_-]{11})/);
  return shorts ? shorts[1] : null;
}

function send(message) {
  // throws if the extension was reloaded and this script is left behind
  try {
    chrome.runtime.sendMessage(message).catch(() => {});
  } catch (e) {
    stopActive();
  }
}

function isVisible(video) {
  const box = video.getBoundingClientRect();
  return box.width > 0 && box.height > 0 && box.bottom > 0 && box.top < window.innerHeight;
}

function findActiveVideo() {
  // on Shorts several players exist at once, the active reel is the one on screen
  const reel = document.querySelector("ytd-reel-video-renderer[is-active] video");
  if (reel) return reel;
  const videos = [...document.querySelectorAll("video")].filter((v) => v.currentSrc && isVisible(v));
  return videos.find((v) => !v.paused) || videos[0] || document.querySelector("#movie_player video");
}

function adIsShowing(video) {
  const player = (video && video.closest(".html5-video-player")) || document.querySelector("#movie_player");
  return Boolean(player && player.classList.contains("ad-showing"));
}

function stopActive() {
  if (!active) return;
  const current = active;
  active = null;
  clearInterval(current.timer);
  if (current.recorder && current.recorder.state !== "inactive") current.recorder.stop();
  if (current.stream) current.stream.getTracks().forEach((track) => track.stop());
}

function fail(videoId) {
  stopActive();
  send({ type: "recordingFailed", videoId });
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function startRecording(videoId) {
  stopActive();
  if (videoIdFromUrl() !== videoId) {
    send({ type: "recordingFailed", videoId });
    return;
  }

  // pre-roll ad. wait for it to end, the ad is not the video being scanned
  let waited = 0;
  active = { videoId, timer: null };
  const check = () => {
    const video = findActiveVideo();
    if (!adIsShowing(video)) {
      clearInterval(active.timer);
      beginCapture(videoId, video);
    } else if (waited >= AD_WAIT_SECONDS) {
      fail(videoId);
    } else {
      waited += 1;
      send({ type: "recordingProgress", videoId, seconds: 0, waiting: "ad" });
    }
  };
  active.timer = setInterval(check, 1000);
  check();
}

function beginCapture(videoId, video) {
  if (!video) {
    fail(videoId);
    return;
  }
  if (video.paused) {
    stopActive();
    send({ type: "recordingPaused", videoId });
    return;
  }

  try {
    const audioTracks = video.captureStream().getAudioTracks();
    if (audioTracks.length === 0) throw new Error("no audio tracks");

    const stream = new MediaStream(audioTracks);
    const recorder = new MediaRecorder(stream, { mimeType: MIME_TYPE });
    const chunks = [];
    const recording = { videoId, recorder, stream, timer: null };
    active = recording;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onerror = () => {
      if (active === recording) fail(videoId);
    };
    recorder.onstop = async () => {
      // stopActive clears active first, so anything else means the clip was thrown away
      if (!recording.finished) return;
      const blob = new Blob(chunks, { type: "audio/webm" });
      if (blob.size === 0) {
        send({ type: "recordingFailed", videoId });
        return;
      }
      try {
        const audioBase64 = await blobToBase64(blob);
        send({ type: "clipReady", videoId, audioBase64, mimeType: "audio/webm" });
      } catch (e) {
        send({ type: "recordingFailed", videoId });
      }
    };

    recorder.start(1000);
    let seconds = 0;
    let held = 0;

    // ticks every second, which also keeps the service worker awake
    recording.timer = setInterval(() => {
      if (video.paused || adIsShowing(video)) {
        // only the video itself should end up in the clip
        if (recorder.state === "recording") recorder.pause();
        held += 1;
        if (held > MAX_HELD_SECONDS) {
          fail(videoId);
          return;
        }
        send({
          type: "recordingProgress",
          videoId,
          seconds,
          waiting: video.paused ? "paused" : "ad",
        });
        return;
      }
      if (recorder.state === "paused") recorder.resume();
      seconds += 1;
      send({ type: "recordingProgress", videoId, seconds });
      if (seconds >= RECORD_SECONDS) {
        recording.finished = true;
        stopActive();
      }
    }, 1000);
  } catch (e) {
    fail(videoId);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "record") startRecording(message.videoId);
  sendResponse({ ok: true });
});

document.addEventListener("yt-navigate-finish", () => {
  const videoId = videoIdFromUrl();
  if (active && active.videoId !== videoId) {
    const cancelled = active.videoId;
    stopActive();
    send({ type: "recordingCancelled", videoId: cancelled });
  }
  send({ type: "navigated", videoId });
});
