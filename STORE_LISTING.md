# Chrome Web Store listing

Text to paste into the developer dashboard. The zip, the fee and the submission are done by hand.

## Short description

Checks how likely the YouTube video you are watching is to be AI generated, with separate voice, visuals and metadata scores.

## Single purpose

Estimate how likely the current YouTube video is to be AI generated.

## Permission justifications

- **activeTab:** Reads the URL of the current tab when the user clicks the extension, to get the YouTube video ID.
- **storage:** Keeps each video's scan progress and result in session storage, so the result is still there when the popup is closed and reopened.
- **Host permission, youtube.com:** Runs the content script on YouTube, which records a short clip of the audio of the video that is playing, after the user opens the extension.
- **Host permission, backend URL:** Sends the video ID and audio clip to the backend, which returns the score.

## Remote code

No. All extension code is in the package.

## Data usage disclosure

- Website content: yes, a short audio clip from the current video and its video ID, sent to compute the score. The clip is deleted right after scoring.
- Not sold, not used for anything unrelated to the extension's purpose, not used for creditworthiness or lending.

## Privacy policy URL

Host `PRIVACY.md` publicly, for example with GitHub Pages, and paste the link here.

## Screenshot

At least one at 1280x800. Take it during extension testing, with the popup open on a video.
