import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ZONE_COLORS = {
    "crowd_space": (60, 180, 75),
    "walls_barriers": (95, 95, 95),
    "ignore_background": (130, 130, 130),
}

LEGACY_ZONE_ALIASES = {
    "zones": ("main_walkable_zone", "upper_right_secondary_zone", "bottleneck_zone"),
    "boundaries": ("left_wall", "right_wall"),
    "ignore_regions": ("sky_horizon", "left_buildings", "far_background_structures"),
}

ANNOTATION_ZONE_NAMES = {
    "zones": "crowd_space",
    "boundaries": "walls_barriers",
    "ignore_regions": "ignore_background",
}


@dataclass
class SceneProfile:
    name: str
    frame_width: int
    frame_height: int
    zones: dict
    boundaries: dict
    ignore_regions: dict
    semantic_weights: dict
    risk_modifiers: dict
    notes: dict
    source_path: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        frame_size = payload.get("frame_size", {})
        return cls(
            name=payload.get("profile_name", path.stem),
            frame_width=int(frame_size.get("width", 1)),
            frame_height=int(frame_size.get("height", 1)),
            zones=payload.get("zones", {}),
            boundaries=payload.get("boundaries", {}),
            ignore_regions=payload.get("ignore_regions", {}),
            semantic_weights=payload.get("semantic_weights", {}),
            risk_modifiers=payload.get("risk_modifiers", {}),
            notes=payload.get("notes", {}),
            source_path=path,
        )

    @classmethod
    def load_for_video(cls, video_path: str | Path, profiles_dir: str | Path):
        video_path = Path(video_path)
        profiles_dir = Path(profiles_dir)
        candidate = profiles_dir / f"{video_path.stem}.json"
        if candidate.exists():
            return cls.from_json(candidate)
        return None

    def _scale_polygon(self, polygon, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        sx = frame_w / max(1, self.frame_width)
        sy = frame_h / max(1, self.frame_height)
        pts = np.array([[int(round(x * sx)), int(round(y * sy))] for x, y in polygon], dtype=np.int32)
        return pts.reshape((-1, 1, 2))

    @staticmethod
    def _is_point_list(value):
        return (
            bool(value)
            and isinstance(value, list)
            and isinstance(value[0], (list, tuple))
            and len(value[0]) == 2
            and isinstance(value[0][0], (int, float))
        )

    def _polygon_list(self, value):
        if not value:
            return []
        if self._is_point_list(value):
            return [value]
        if isinstance(value, list):
            polygons = []
            for polygon in value:
                if self._is_point_list(polygon):
                    polygons.append(polygon)
            return polygons
        return []

    def _category_polygons(self, group_name: str):
        source = getattr(self, group_name)
        category_name = ANNOTATION_ZONE_NAMES[group_name]
        polygons = []

        if category_name in source:
            polygons.extend(self._polygon_list(source[category_name]))
        else:
            for legacy_name in LEGACY_ZONE_ALIASES[group_name]:
                if legacy_name in source:
                    polygons.extend(self._polygon_list(source[legacy_name]))
            if not polygons:
                for value in source.values():
                    polygons.extend(self._polygon_list(value))

        return polygons

    def export_annotation_group(self, group_name: str):
        category_name = ANNOTATION_ZONE_NAMES[group_name]
        polygons = self._category_polygons(group_name)
        if not polygons:
            return {}
        return {category_name: polygons[0]}

    def create_mask(self, polygons, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        polygon_list = []
        if isinstance(polygons, dict):
            for polygon in polygons.values():
                polygon_list.extend(self._polygon_list(polygon))
        else:
            polygon_list.extend(self._polygon_list(polygons))
        for polygon in polygon_list:
            pts = self._scale_polygon(polygon, frame_shape)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def get_ignore_mask(self, frame_shape):
        polygons = self._category_polygons("ignore_regions")
        if not polygons:
            return np.zeros(frame_shape[:2], dtype=np.uint8)
        return self.create_mask(polygons, frame_shape)

    def get_walkable_mask(self, frame_shape):
        polygons = self._category_polygons("zones")
        if polygons:
            return self.create_mask(polygons, frame_shape)
        return np.full(frame_shape[:2], 255, dtype=np.uint8)

    def get_boundary_mask(self, frame_shape):
        polygons = self._category_polygons("boundaries")
        if not polygons:
            return np.zeros(frame_shape[:2], dtype=np.uint8)
        return self.create_mask(polygons, frame_shape)

    def get_spatial_analysis_mask(self, frame_shape):
        analysis = self.get_walkable_mask(frame_shape)
        ignore = self.get_ignore_mask(frame_shape)
        boundaries = self.get_boundary_mask(frame_shape)
        analysis[ignore > 0] = 0
        analysis[boundaries > 0] = 0
        return analysis

    def render_overlay(self, frame, alpha: float = 0.10):
        overlay = frame.copy()

        for group_name in ("zones", "boundaries", "ignore_regions"):
            category_name = ANNOTATION_ZONE_NAMES[group_name]
            color = ZONE_COLORS[category_name]
            polygons = self._category_polygons(group_name)
            if not polygons:
                continue

            label_points = []
            for polygon in polygons:
                pts = self._scale_polygon(polygon, frame.shape)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
                label_points.append(pts.reshape(-1, 2).mean(axis=0))

            label_origin = np.mean(np.array(label_points), axis=0).astype(int)
            cv2.putText(
                overlay,
                category_name.replace("_", " ").title(),
                (int(label_origin[0]) - 55, int(label_origin[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0.0)
