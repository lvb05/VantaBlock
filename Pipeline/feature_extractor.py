import json
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class FeatureExtractor:
    def load_segment(self, filepath: Path) -> Dict[str, Any]:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)

    def time_to_first_action(self, events: List[Dict[str, Any]]) -> float:
        if not events:
            return 0.0
        start_ts = events[0].get("timestamp_ms", 0)
        for event in events:
            if event.get("type") in {"mousedown", "mouseup", "click", "keydown", "scroll", "navigation"}:
                return float(event.get("timestamp_ms", start_ts) - start_ts)
        return 0.0

    def inter_event_std(self, events: List[Dict[str, Any]]) -> float:
        if len(events) < 2:
            return 0.0
        timestamps = [e.get("timestamp_ms", 0) for e in events]
        deltas = np.diff(timestamps)
        return float(np.std(deltas))

    def path_efficiency(self, events: List[Dict[str, Any]]) -> float:
        efficiencies: List[float] = []
        for event in events:
            if event.get("type") != "mousemove":
                continue
            path = event.get("data", {}).get("path", [])
            if len(path) < 2:
                continue

            actual = 0.0
            for i in range(1, len(path)):
                dx = float(path[i].get("x", 0) - path[i - 1].get("x", 0))
                dy = float(path[i].get("y", 0) - path[i - 1].get("y", 0))
                actual += float(np.hypot(dx, dy))

            start = path[0]
            end = path[-1]
            straight = float(
                np.hypot(
                    float(end.get("x", 0) - start.get("x", 0)),
                    float(end.get("y", 0) - start.get("y", 0)),
                )
            )

            if actual > 0:
                efficiencies.append(straight / actual)

        if not efficiencies:
            return 0.0
        return float(np.mean(efficiencies))

    def velocity_variance(self, events: List[Dict[str, Any]]) -> float:
        velocities: List[float] = []
        for event in events:
            if event.get("type") != "mousemove":
                continue
            path = event.get("data", {}).get("path", [])
            if len(path) < 2:
                continue

            for i in range(1, len(path)):
                dx = float(path[i].get("x", 0) - path[i - 1].get("x", 0))
                dy = float(path[i].get("y", 0) - path[i - 1].get("y", 0))
                dt = float(path[i].get("t_ms", 0) - path[i - 1].get("t_ms", 0))
                if dt <= 0:
                    continue
                dist = float(np.hypot(dx, dy))
                velocities.append(dist / dt)

        if len(velocities) < 2:
            return 0.0
        return float(np.var(velocities))

    def hover_time_before_click(self, events: List[Dict[str, Any]]) -> float:
        hover_times: List[float] = []
        for i, event in enumerate(events):
            if event.get("type") != "click":
                continue

            click_ts = event.get("timestamp_ms", 0)
            click_target = event.get("data", {}).get("target_element")
            if click_target is None:
                continue

            first_same_target_ts = None
            for j in range(i - 1, -1, -1):
                prev = events[j]
                prev_type = prev.get("type")

                if prev_type == "mousemove":
                    prev_target = prev.get("data", {}).get("target_element")
                    if prev_target == click_target:
                        first_same_target_ts = prev.get("timestamp_ms", click_ts)
                        continue
                    break

                if prev_type in {"click", "mousedown", "mouseup", "keydown", "scroll", "navigation"}:
                    break

            if first_same_target_ts is not None:
                hover_times.append(float(click_ts - first_same_target_ts))

        if not hover_times:
            return 0.0
        return float(np.mean(hover_times))

    def scroll_variance(self, events: List[Dict[str, Any]]) -> float:
        steps: List[float] = []
        for event in events:
            if event.get("type") != "scroll":
                continue
            data = event.get("data", {})
            if "deltaX" in data or "deltaY" in data:
                dx = float(data.get("deltaX", 0))
                dy = float(data.get("deltaY", 0))
                steps.append(float(np.hypot(dx, dy)))
                continue

            scroll_steps = data.get("steps", [])
            if len(scroll_steps) < 2:
                continue
            for i in range(1, len(scroll_steps)):
                dx = float(scroll_steps[i].get("x", 0) - scroll_steps[i - 1].get("x", 0))
                dy = float(scroll_steps[i].get("y", 0) - scroll_steps[i - 1].get("y", 0))
                steps.append(float(np.hypot(dx, dy)))

        if len(steps) < 2:
            return 0.0
        return float(np.var(steps))

    def error_behavior(self, events: List[Dict[str, Any]]) -> float:
        keydown_events = [e for e in events if e.get("type") == "keydown"]
        if not keydown_events:
            return 0.0
        backspaces = sum(1 for e in keydown_events if e.get("data", {}).get("key") == "Backspace")
        return float(backspaces / len(keydown_events))

    def extract_features_from_segment(self, segment_data: Dict[str, Any]) -> Dict[str, Any]:
        events = segment_data.get("events", [])
        if not events:
            return {}

        raw_label = str(segment_data.get("label") or segment_data.get("session_type") or "").lower()
        if raw_label not in {"human", "agent"}:
            return {}

        is_human = 1 if raw_label == "human" else 0

        return {
            "segment_id": segment_data.get("segment_id") or segment_data.get("session_id") or "",
            "is_human": is_human,
            "time_to_first_action": self.time_to_first_action(events),
            "inter_event_std": self.inter_event_std(events),
            "path_efficiency": self.path_efficiency(events),
            "velocity_variance": self.velocity_variance(events),
            "hover_time_before_click": self.hover_time_before_click(events),
            "scroll_variance": self.scroll_variance(events),
            "error_behavior": self.error_behavior(events),
        }

    def process_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        rows: List[Dict[str, Any]] = []
        segment_files = sorted(dataset_dir.glob("segment_*.json"))
        files = segment_files if segment_files else sorted(dataset_dir.glob("*.json"))
        for filepath in files:
            segment_data = self.load_segment(filepath)
            row = self.extract_features_from_segment(segment_data)
            if row:
                rows.append(row)

        return rows

    def process_multiple_datasets(self, dataset_paths: List[str]) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        for path in dataset_paths:
            all_rows.extend(self.process_dataset(path))
        return all_rows


def main() -> None:
    human_dataset_path = r"C:\Users\indra\Desktop\Tuta hua gulab jamun\mockshop\mockshop\Mentor_round\human"
    agent_dataset_path = r"C:\Users\indra\Desktop\Tuta hua gulab jamun\mockshop\mockshop\Mentor_round\Agent"
    output_dir = Path(r"C:\Users\indra\Desktop\Tuta hua gulab jamun\mockshop\mockshop\Mentor_round")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "features.csv"

    extractor = FeatureExtractor()
    rows = extractor.process_multiple_datasets([human_dataset_path, agent_dataset_path])

    fieldnames = [
        "segment_id",
        "is_human",
        "time_to_first_action",
        "inter_event_std",
        "path_efficiency",
        "velocity_variance",
        "hover_time_before_click",
        "scroll_variance",
        "error_behavior",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
