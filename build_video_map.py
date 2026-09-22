"""
Convert yt-dlp flat-playlist output into video-map.json
(matches the shape expected by the ytscrapper frontend).

Usage:
    yt-dlp --flat-playlist --print "%(playlist_index)s|%(id)s|%(title)s" "<PLAYLIST_URL>" > video_map_raw.txt
    python build_video_map.py video_map_raw.txt video-map.json
"""
import json
import sys

def build(raw_path: str, out_path: str) -> None:
    number_to_id = {}
    title_to_id = {}

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                print(f"Skipping malformed line: {line!r}", file=sys.stderr)
                continue
            index, video_id, title = parts
            number_to_id[index] = video_id
            title_to_id[title] = video_id

    result = dict(number_to_id)
    result["title_to_id"] = title_to_id

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(number_to_id)} mappings to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_video_map.py <raw_txt> <out_json>", file=sys.stderr)
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])