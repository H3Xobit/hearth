"""RTMPose through rtmlib. Imported only when someone runs a video.

Balanced mode is RTMPose-m. Performance mode is the larger RTMPose that
rtmlib selects for accuracy. Both are Apache-2.0 weights from OpenMMLab.
"""

from __future__ import annotations


class PoseReader:
    def __init__(self, mode: str = "performance", device: str = "cpu"):
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise RuntimeError(
                "Pose weights are not installed. Use: pip install -e '.[pose]'"
            ) from exc
        self._body = Body(
            mode=mode,
            backend="onnxruntime",
            device=device,
            to_openpose=False,
        )

    def people(self, frame):
        """Return a list of (17, 3) COCO bodies, highest confidence first."""
        keypoints, scores = self._body(frame)
        if keypoints is None or len(keypoints) == 0:
            return []
        bodies = []
        for person, conf in zip(keypoints, scores):
            points = [
                (float(person[i][0]), float(person[i][1]), float(conf[i]))
                for i in range(min(17, len(person)))
            ]
            mean = sum(p[2] for p in points) / len(points)
            bodies.append((mean, points))
        bodies.sort(key=lambda item: item[0], reverse=True)
        return [points for _, points in bodies]
