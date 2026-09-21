import json
from pathlib import Path

import yt_dlp


def download_video_transcript(video_url: str, output_dir: str = "downloads") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "outtmpl": str(output_path / "%(title)s.%(ext)s"),
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return str(output_path)


if __name__ == "__main__":
    sample_url = "https://www.youtube.com/watch?v=example"
    print(download_video_transcript(sample_url))
