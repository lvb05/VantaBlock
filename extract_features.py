import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, variance


def safe_mean(values, default=0.0):
    return float(mean(values)) if values else float(default)


def safe_var(values, default=0.0):
    return float(variance(values)) if len(values) > 1 else float(default)


def angle_diff(a, b):
    d = abs(a - b) % (2 * math.pi)
    return d if d <= math.pi else (2 * math.pi - d)


def parse_events(session):
    events = session.get("events", [])
    events = [e for e in events if isinstance(e, dict) and "timestamp_ms" in e and "type" in e]
    events.sort(key=lambda x: x.get("timestamp_ms", 0))
    return events


def extract_mouse_points(events):
    points = []
    last_key = None
    for e in events:
        if e.get("type") != "mousemove":
            continue
        path = e.get("data", {}).get("path", [])
        if not isinstance(path, list):
            continue
        for p in path:
            if not isinstance(p, dict):
                continue
            x = p.get("x")
            y = p.get("y")
            t = p.get("t_ms")
            if x is None or y is None or t is None:
                continue
            key = (int(x), int(y), int(t))
            if key == last_key:
                continue
            points.append({"x": float(x), "y": float(y), "t_ms": int(t)})
            last_key = key
    points.sort(key=lambda p: p["t_ms"])
    return points


def mouse_metrics(points):
    if len(points) < 2:
        return {
            "mouse_path_length": 0.0,
            "mouse_duration_ms": 0.0,
            "mouse_avg_velocity": 0.0,
            "mouse_velocity_var": 0.0,
            "mouse_peak_velocity": 0.0,
            "mouse_acceleration_var": 0.0,
            "mouse_jerk_cost": 0.0,
            "mouse_direction_changes": 0.0,
            "mouse_straightness": 0.0,
            "mouse_time_to_peak_velocity_ms": 0.0,
        }

    dists = []
    dts = []
    velocities = []
    angles = []
    vel_times = []

    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        dt = points[i]["t_ms"] - points[i - 1]["t_ms"]
        if dt <= 0:
            continue
        dist = math.hypot(dx, dy)
        vel = dist / dt

        dists.append(dist)
        dts.append(dt)
        velocities.append(vel)
        vel_times.append(points[i]["t_ms"])
        if dist > 0:
            angles.append(math.atan2(dy, dx))

    if not velocities:
        return {
            "mouse_path_length": 0.0,
            "mouse_duration_ms": 0.0,
            "mouse_avg_velocity": 0.0,
            "mouse_velocity_var": 0.0,
            "mouse_peak_velocity": 0.0,
            "mouse_acceleration_var": 0.0,
            "mouse_jerk_cost": 0.0,
            "mouse_direction_changes": 0.0,
            "mouse_straightness": 0.0,
            "mouse_time_to_peak_velocity_ms": 0.0,
        }

    accels = []
    for i in range(1, len(velocities)):
        dt = dts[i]
        if dt > 0:
            accels.append((velocities[i] - velocities[i - 1]) / dt)

    jerks = []
    for i in range(1, len(accels)):
        dt = dts[i + 1] if (i + 1) < len(dts) else 0
        if dt > 0:
            jerks.append((accels[i] - accels[i - 1]) / dt)

    direction_changes = 0
    for i in range(1, len(angles)):
        if angle_diff(angles[i], angles[i - 1]) > (math.pi / 6):
            direction_changes += 1

    start = points[0]
    end = points[-1]
    displacement = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
    path_length = sum(dists)
    straightness = (displacement / path_length) if path_length > 0 else 0.0

    peak_velocity = max(velocities)
    peak_idx = velocities.index(peak_velocity)
    time_to_peak = max(0, vel_times[peak_idx] - points[0]["t_ms"])

    return {
        "mouse_path_length": path_length,
        "mouse_duration_ms": max(0, points[-1]["t_ms"] - points[0]["t_ms"]),
        "mouse_avg_velocity": safe_mean(velocities),
        "mouse_velocity_var": safe_var(velocities),
        "mouse_peak_velocity": peak_velocity,
        "mouse_acceleration_var": safe_var(accels),
        "mouse_jerk_cost": safe_mean([abs(j) for j in jerks]),
        "mouse_direction_changes": float(direction_changes),
        "mouse_straightness": straightness,
        "mouse_time_to_peak_velocity_ms": float(time_to_peak),
    }


def click_metrics(events):
    clicks = [e for e in events if e.get("type") == "click"]
    mouseups = [e for e in events if e.get("type") == "mouseup"]
    mousemoves = [e for e in events if e.get("type") == "mousemove"]

    offsets_x = []
    offsets_y = []
    for c in clicks:
        off = c.get("data", {}).get("offset_from_center", {})
        offsets_x.append(float(off.get("dx", 0)))
        offsets_y.append(float(off.get("dy", 0)))

    holds = [
        float(e.get("data", {}).get("hold_duration_ms", 0))
        for e in mouseups
        if e.get("data", {}).get("hold_duration_ms") is not None
    ]

    move_timestamps = []
    for m in mousemoves:
        path = m.get("data", {}).get("path", [])
        if isinstance(path, list):
            for p in path:
                if isinstance(p, dict) and p.get("t_ms") is not None:
                    move_timestamps.append(int(p["t_ms"]))
    move_timestamps.sort()

    event_times = [int(e.get("timestamp_ms", 0)) for e in events]
    click_times = [int(c.get("timestamp_ms", 0)) for c in clicks]

    pre_hover = []
    for ct in click_times:
        prior = [mt for mt in move_timestamps if mt <= ct]
        if prior:
            pre_hover.append(max(0, ct - prior[-1]))

    post_dwell = []
    for ct in click_times:
        nxt = [t for t in event_times if t > ct]
        if nxt:
            post_dwell.append(max(0, nxt[0] - ct))

    intervals = []
    for i in range(1, len(click_times)):
        intervals.append(max(0, click_times[i] - click_times[i - 1]))

    return {
        "click_count": float(len(clicks)),
        "click_offset_x_mean": safe_mean(offsets_x),
        "click_offset_y_mean": safe_mean(offsets_y),
        "click_offset_abs_mean": safe_mean([math.hypot(dx, dy) for dx, dy in zip(offsets_x, offsets_y)]),
        "click_hold_mean_ms": safe_mean(holds),
        "click_hold_var_ms": safe_var(holds),
        "pre_click_hover_mean_ms": safe_mean(pre_hover),
        "post_click_dwell_mean_ms": safe_mean(post_dwell),
        "click_interval_mean_ms": safe_mean(intervals),
        "click_interval_var_ms": safe_var(intervals),
    }


def scroll_event_stats(scroll_event):
    data = scroll_event.get("data", {})
    start_y = float(data.get("start_y", 0))
    end_y = float(data.get("end_y", start_y))
    steps = data.get("steps", []) if isinstance(data.get("steps", []), list) else []

    ys = [start_y] + [float(s.get("y", start_y)) for s in steps]
    ts = [scroll_event.get("timestamp_ms", 0)] + [int(s.get("t_ms", scroll_event.get("timestamp_ms", 0))) for s in steps]

    deltas = []
    intervals = []
    velocities = []
    for i in range(1, len(ys)):
        dy = ys[i] - ys[i - 1]
        dt = ts[i] - ts[i - 1]
        deltas.append(dy)
        if dt > 0:
            intervals.append(dt)
            velocities.append(dy / dt)

    primary = end_y - start_y
    if primary == 0:
        primary = next((d for d in deltas if d != 0), 1)
    primary_sign = 1 if primary >= 0 else -1

    corrections = [d for d in deltas if d != 0 and (1 if d >= 0 else -1) != primary_sign]
    overshoot = sum(abs(d) for d in corrections)

    return {
        "step_count": float(len(steps)),
        "step_intervals": intervals,
        "step_velocities": velocities,
        "overshoot": overshoot,
        "has_correction": 1.0 if corrections else 0.0,
    }


def scroll_metrics(events):
    scrolls = [e for e in events if e.get("type") == "scroll"]
    per_event = [scroll_event_stats(s) for s in scrolls]

    step_counts = [x["step_count"] for x in per_event]
    overshoots = [x["overshoot"] for x in per_event]
    corrections = [x["has_correction"] for x in per_event]

    intervals = []
    velocities = []
    for x in per_event:
        intervals.extend(x["step_intervals"])
        velocities.extend(x["step_velocities"])

    return {
        "scroll_event_count": float(len(scrolls)),
        "scroll_step_count_mean": safe_mean(step_counts),
        "scroll_step_interval_var_ms": safe_var(intervals),
        "scroll_velocity_var": safe_var(velocities),
        "scroll_overshoot_mean": safe_mean(overshoots),
        "scroll_correction_rate": safe_mean(corrections),
    }


def typing_metrics(events):
    keydowns = [e for e in events if e.get("type") == "keydown"]
    keyups = [e for e in events if e.get("type") == "keyup"]

    kd_times = [int(e.get("timestamp_ms", 0)) for e in keydowns]
    key_intervals = [
        max(0, kd_times[i] - kd_times[i - 1])
        for i in range(1, len(kd_times))
    ]

    hold_durations = [
        float(e.get("data", {}).get("hold_duration_ms", 0))
        for e in keyups
        if e.get("data", {}).get("hold_duration_ms") is not None
    ]

    backspaces = [
        e for e in keydowns if str(e.get("data", {}).get("key", "")).lower() == "backspace"
    ]

    pause_correct = 0
    for i in range(1, len(keydowns)):
        key = str(keydowns[i].get("data", {}).get("key", ""))
        gap = keydowns[i].get("timestamp_ms", 0) - keydowns[i - 1].get("timestamp_ms", 0)
        if gap > 800 and key.lower() == "backspace":
            pause_correct += 1

    total_keydowns = len(keydowns)
    backspace_freq = (len(backspaces) / total_keydowns) if total_keydowns > 0 else 0.0

    return {
        "keystroke_count": float(total_keydowns),
        "inter_key_interval_mean_ms": safe_mean(key_intervals),
        "inter_key_interval_var_ms": safe_var(key_intervals),
        "key_hold_mean_ms": safe_mean(hold_durations),
        "key_hold_var_ms": safe_var(hold_durations),
        "backspace_frequency": backspace_freq,
        "pause_correct_events": float(pause_correct),
    }


def session_metrics(session, events):
    timestamps = [int(e.get("timestamp_ms", 0)) for e in events]
    gaps = [
        max(0, timestamps[i] - timestamps[i - 1])
        for i in range(1, len(timestamps))
    ]

    first_interaction = 0
    for e in events:
        if e.get("type") != "navigation":
            first_interaction = int(e.get("timestamp_ms", 0))
            break

    counts = {}
    for e in events:
        t = str(e.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1

    return {
        "session_duration_ms": float(session.get("end_time_ms", 0)),
        "time_to_first_interaction_ms": float(first_interaction),
        "inter_event_gap_mean_ms": safe_mean(gaps),
        "inter_event_gap_var_ms": safe_var(gaps),
        "event_count_total": float(len(events)),
        "event_count_navigation": float(counts.get("navigation", 0)),
        "event_count_mousemove": float(counts.get("mousemove", 0)),
        "event_count_mousedown": float(counts.get("mousedown", 0)),
        "event_count_mouseup": float(counts.get("mouseup", 0)),
        "event_count_click": float(counts.get("click", 0)),
        "event_count_scroll": float(counts.get("scroll", 0)),
        "event_count_keydown": float(counts.get("keydown", 0)),
        "event_count_keyup": float(counts.get("keyup", 0)),
    }


def extract_row(session):
    events = parse_events(session)
    mouse_points = extract_mouse_points(events)

    row = {
        "session_id": session.get("session_id", "unknown"),
        "session_type": session.get("session_type", "unknown"),
    }
    row.update(mouse_metrics(mouse_points))
    row.update(click_metrics(events))
    row.update(scroll_metrics(events))
    row.update(typing_metrics(events))
    row.update(session_metrics(session, events))
    return row


def load_sessions(log_dir):
    sessions = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            sessions.append(data)
    return sessions


def write_csv(rows, output_path):
    if not rows:
        print("No valid session logs found.")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved feature table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ML-ready features from unified behavioural session logs")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing session JSON files")
    parser.add_argument("--output", default="features.csv", help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.logs_dir)
    rows = [extract_row(s) for s in load_sessions(log_dir)]
    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
