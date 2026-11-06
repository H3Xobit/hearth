"""Read a video file at 4 frames a second and print one JSON object per sample.

    python -m roomwatch --video clip.mp4 --room lounge
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank posture in a video file.")
    parser.add_argument("--video", required=True, help="Path to a video file.")
    parser.add_argument("--room", default="room", help="Name put on each alert message.")
    parser.add_argument("--fps", type=float, default=4.0, help="Samples per second.")
    parser.add_argument(
        "--mode",
        default="performance",
        choices=("lightweight", "balanced", "performance"),
        help="RTMPose size. performance is the accuracy choice.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    if args.fps <= 0:
        print("fps must be positive", file=sys.stderr)
        return 2
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Use: pip install -e '.[pose]'", file=sys.stderr)
        return 2
    from roomwatch.pose import PoseReader
    from roomwatch.session import Session

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"Could not open {args.video}", file=sys.stderr)
        return 2
    reader = PoseReader(mode=args.mode, device=args.device)
    session = Session(args.room)
    step = 1.0 / args.fps
    next_t = 0.0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = index / max(capture.get(cv2.CAP_PROP_FPS) or args.fps, 1.0)
        index += 1
        if timestamp + 1e-6 < next_t:
            continue
        next_t = timestamp + step
        people = reader.people(frame)
        points = people[0] if people else None
        view = session.update(timestamp, points)
        row = {
            "t": round(timestamp, 3),
            "leader": view["leader"],
            "state": view["state"],
            "ranked": [(name, round(score, 4)) for name, score in view["ranked"]],
        }
        if view["message"] is not None:
            row["message"] = view["message"]
        print(json.dumps(row))
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
