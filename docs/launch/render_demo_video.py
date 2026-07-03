"""Render the launch terminal demo to MP4/GIF using ffmpeg.

This avoids external terminal-recording tools. It runs
``examples/launch_demo.py --no-pause``, turns the output into a terminal-style
ASS subtitle track, and renders:

- ``docs/launch/synaptic-memory-demo.mp4``
- ``docs/launch/synaptic-memory-demo.gif``

Requires ``ffmpeg`` with ``ass``, ``drawtext``, ``palettegen``, and
``paletteuse`` filters.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = ROOT / "docs" / "launch"
MP4_PATH = LAUNCH_DIR / "synaptic-memory-demo.mp4"
GIF_PATH = LAUNCH_DIR / "synaptic-memory-demo.gif"
ASS_PATH = LAUNCH_DIR / "synaptic-memory-demo.ass"
PALETTE_PATH = LAUNCH_DIR / "synaptic-memory-demo-palette.png"
FONT_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf")


SCENES = [
    {
        "title": "Synaptic Memory",
        "lines": [
            "Graph memory for RAG agents",
            "Documents + SQL rows + feedback",
            "Default path: no LLM calls at indexing time",
        ],
        "hold": 1.0,
    },
    {
        "title": "1. Build a tiny mixed graph",
        "lines": [
            "Created graph: 4 nodes",
            "Sources: 2 policy docs + 2 support ticket rows",
        ],
        "hold": 1.2,
    },
    {
        "title": "2. Search and record retrieval",
        "lines": [
            "event_id: <retrieval-event-id>",
            "1. ticket:T-1001                score=0.660 source=ticket",
            "2. Refund Policy                score=0.435 source=policy/refunds.md",
            "3. Shipping Policy              score=0.210 source=policy/shipping.md",
        ],
        "hold": 1.2,
    },
    {
        "title": "3. Feed back that the evidence helped",
        "lines": [
            "Recorded explicit positive feedback for the top evidence.",
        ],
        "hold": 1.2,
    },
    {
        "title": "4. Inspect memory health metadata",
        "lines": [
            "memory_events:    2",
            "retrieval_events: 2",
            "memory_scores:    2",
            "health_signals:   0",
            "No raw provenance was appended to Node.content.",
            "Swap the backend to PostgreSQL/Kuzu/Qdrant when the corpus grows.",
        ],
        "hold": 2.0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4-only", action="store_true", help="Skip GIF export.")
    args = parser.parse_args()

    ffmpeg = _require("ffmpeg")
    uv = _require("uv")
    _validate_demo_output(uv)

    duration = _write_ass(ASS_PATH)
    _render_mp4(duration, ffmpeg)
    if not args.mp4_only:
        _render_gif(ffmpeg)
    print(f"rendered: {MP4_PATH}")
    if not args.mp4_only:
        print(f"rendered: {GIF_PATH}")


def _require(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"missing required command: {name}")
    return resolved


def _validate_demo_output(uv: str) -> None:
    result = subprocess.run(  # noqa: S603 - fixed command for local launch rendering.
        [uv, "run", "python", "examples/launch_demo.py", "--no-pause"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    normalized = re.sub(r"event_id: [0-9a-f]+", "event_id: <retrieval-event-id>", result.stdout)
    required = [
        "Created graph: 4 nodes",
        "ticket:T-1001",
        "Refund Policy",
        "Recorded explicit positive feedback",
        "memory_scores:   2",
    ]
    missing = [needle for needle in required if needle not in normalized]
    if missing:
        raise SystemExit(f"demo output missing expected lines: {missing}")


def _write_ass(path: Path) -> float:
    events: list[str] = []
    t = 0.0
    for scene in SCENES:
        visible = [scene["title"]]
        for line in scene["lines"]:
            start = t
            t += 1.0
            visible.append(line)
            events.append(_dialogue(start, t, visible))
        end = t + float(scene["hold"])
        events[-1] = _dialogue(t - 1.0, end, visible)
        t = end

    content = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1280",
            "PlayResY: 720",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            "Style: Terminal,Liberation Mono,30,&H00E5E7EB,&H000000FF,&H00111827,"
            "&H00111827,0,0,0,0,100,100,0,0,1,0,0,7,80,60,95,1",
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
            *events,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return t


def _dialogue(start: float, end: float, lines: list[str]) -> str:
    text = r"\N".join(_escape_ass(line) for line in lines)
    return f"Dialogue: 0,{_ts(start)},{_ts(end)},Terminal,,0,0,0,,{text}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _ts(seconds: float) -> str:
    centis = round(seconds * 100)
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _render_mp4(duration: float, ffmpeg: str) -> None:
    font = FONT_PATH
    ass = _ffmpeg_path(ASS_PATH)
    draw_filter = (
        "drawbox=x=44:y=42:w=1192:h=636:color=0x111827@0.96:t=fill,"
        "drawbox=x=44:y=42:w=1192:h=636:color=0x334155@1:t=2,"
        "drawbox=x=44:y=42:w=1192:h=42:color=0x1f2937@1:t=fill,"
        f"drawtext=fontfile='{font}':text='Synaptic Memory launch demo':"
        "x=74:y=53:fontsize=20:fontcolor=0xCBD5E1,"
        f"ass='{ass}'"
    )
    subprocess.run(  # noqa: S603 - fixed ffmpeg invocation over repo-local generated inputs.
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x070B12:s=1280x720:r=30:d={duration:.2f}",
            "-vf",
            draw_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(MP4_PATH),
        ],
        cwd=ROOT,
        check=True,
    )


def _render_gif(ffmpeg: str) -> None:
    scale = "fps=12,scale=960:-1:flags=lanczos"
    subprocess.run(  # noqa: S603 - fixed ffmpeg invocation over generated MP4.
        [
            ffmpeg,
            "-y",
            "-i",
            str(MP4_PATH),
            "-vf",
            f"{scale},palettegen",
            str(PALETTE_PATH),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(  # noqa: S603 - fixed ffmpeg invocation over generated MP4/palette.
        [
            ffmpeg,
            "-y",
            "-i",
            str(MP4_PATH),
            "-i",
            str(PALETTE_PATH),
            "-lavfi",
            f"{scale} [x]; [x][1:v] paletteuse",
            str(GIF_PATH),
        ],
        cwd=ROOT,
        check=True,
    )
    PALETTE_PATH.unlink(missing_ok=True)


def _ffmpeg_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


if __name__ == "__main__":
    main()
