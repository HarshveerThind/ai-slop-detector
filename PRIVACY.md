# Privacy Policy

AI Slop Detector is a Chrome extension that estimates how likely a YouTube video is to be AI generated.

## What the extension sends

When you open the extension on a YouTube video, it sends these to our backend to compute the score:

- The ID of the current video
- A short audio clip, about 20 seconds, recorded from the current video as it plays

The clip is only recorded after you open the extension on a video, and only if no saved score exists for that video. It comes from the video's audio, not your microphone.

## What is stored

- **On our backend:** only the score, keyed by video ID, for up to 7 days. The audio clip is deleted as soon as the score is computed.
- **In your browser:** the scores for videos you scanned, until you close the browser.
- **In server logs:** each request's method, path, status and duration. The path can contain a video ID. Logs do not contain IP addresses.

## What is not collected

We do not collect personal data. There are no accounts. We do not collect your name, email, browsing history or watch history, and scans are not linked to you.

## IP addresses

Your IP address is held in memory for about a minute to limit how many requests one address can make. It is not written to disk or to logs, and it is not used for anything else.

## Other services

The backend looks up the video's public details and images using the YouTube Data API, so Google receives the video ID. It does not send Google anything about you.

We do not sell or share data with anyone else.

## Contact

Questions can be raised as an issue at https://github.com/HarshveerThind/ai-slop-detector/issues.
