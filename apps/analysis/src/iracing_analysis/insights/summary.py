"""Build the structured JSON summary handed to the LLM (or exported)."""

from __future__ import annotations

from ..analysis.corners import analyze_corners, rank_by_time_lost
from ..analysis.consistency import cross_lap_variability, lap_time_stats
from ..analysis.gg import gg_points, gg_stats
from ..analysis.sectors import official_sector_boundaries, sector_times
from ..analysis.theoretical import theoretical_best
from ..analysis.workspace import Workspace

__all__ = ["build_summary"]


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def build_summary(workspace: Workspace, lap_index: int, minisectors: int = 20) -> dict:
    """Everything a coach needs, as plain JSON-able data (SI units + km/h)."""
    meta = workspace.laps[0].meta
    lap = workspace.aligned.laps[lap_index]
    ref = workspace.reference

    comparisons = rank_by_time_lost(analyze_corners(workspace, lap_index))
    corners = [
        {
            "corner": comp.corner.label(),
            "apex_at_m": _r(comp.corner.apex_m, 0),
            "time_lost_s": _r(comp.time_lost),
            "entry_delta_s": _r(comp.delta_entry),
            "exit_delta_s": _r(comp.delta_exit),
            "next_straight_delta_s": _r(comp.delta_carry),
            "braking_point_m": _r(comp.metrics.braking_point_m, 1),
            "ref_braking_point_m": _r(comp.ref_metrics.braking_point_m, 1),
            "min_speed_kmh": _r(comp.metrics.min_speed * 3.6, 1),
            "ref_min_speed_kmh": _r(comp.ref_metrics.min_speed * 3.6, 1),
            "full_throttle_at_m": _r(comp.metrics.throttle_full_m, 1),
            "ref_full_throttle_at_m": _r(comp.ref_metrics.throttle_full_m, 1),
            "trail_brake_overlap_m": _r(comp.metrics.trail_brake_overlap_m, 1),
            "ref_trail_brake_overlap_m": _r(comp.ref_metrics.trail_brake_overlap_m, 1),
            "why": comp.reasons,
        }
        for comp in comparisons
    ]

    sectors = sector_times(workspace, official_sector_boundaries(workspace))
    tb = theoretical_best(workspace, minisectors=minisectors)
    gg = gg_stats(gg_points(lap))
    variability = cross_lap_variability(workspace).summary()

    # tyres & conditions — only what the telemetry actually contains
    signals: dict = {"tyre_data_available": False}
    ch = lap.channels
    temps = {w: f"{w}tempCM" for w in ("LF", "RF", "LR", "RR")}
    if all(name in ch for name in temps.values()):
        signals["tyre_data_available"] = True
        signals["tyre_temp_mid_c_avg"] = {
            w: _r(float(ch[name].mean()), 1) for w, name in temps.items()
        }
    pressures = {w: f"{w}pressure" for w in ("LF", "RF", "LR", "RR")}
    if all(name in ch for name in pressures.values()):
        signals["tyre_pressure_kpa_avg"] = {
            w: _r(float(ch[name].mean()), 1) for w, name in pressures.items()
        }
    if "TrackTempCrew" in ch:
        signals["track_temp_c"] = _r(float(ch["TrackTempCrew"].mean()), 1)
    if "AirTemp" in ch:
        signals["air_temp_c"] = _r(float(ch["AirTemp"].mean()), 1)

    analysed_meta = lap.lap.meta or meta
    return {
        "session": {
            "track": (meta.track_display_name or meta.track_name) if meta else "",
            "track_config": meta.track_config if meta else "",
            "car": meta.car_screen_name if meta else "",
            "driver": analysed_meta.driver_name if analysed_meta else "",
            "track_length_m": _r(meta.track_length_m, 0) if meta else None,
        },
        "analysed_lap": {
            "label": lap.label(),
            "driver": lap.lap.meta.driver_name if lap.lap.meta else "",
            "lap_time_s": _r(lap.lap.lap_time),
        },
        "reference_lap": {
            "label": ref.label(),
            "driver": ref.lap.meta.driver_name if ref.lap.meta else "",
            "lap_time_s": _r(ref.lap.lap_time),
        },
        "total_delta_s": _r(float(workspace.deltas[lap_index][-1])),
        "corners": corners,
        "sectors": {
            "boundaries_m": [_r(b, 0) for b in sectors.boundaries_m],
            "times_s": {
                label: [_r(t) for t in sectors.times[i]]
                for i, label in enumerate(sectors.lap_labels)
            },
        },
        "theoretical_best": {
            "time_s": _r(tb.total),
            "actual_best_s": _r(tb.actual_best),
            "gap_to_actual_best_s": _r(tb.gap),
            "minisector_count": tb.minisector_count,
            "caveat": "optimistic upper bound; mixes mini-sectors from different laps",
        },
        "grip_usage_g": {k: _r(v, 2) for k, v in gg.items()},
        "signals": signals,
        "consistency": {
            "lap_times": {k: _r(v) for k, v in lap_time_stats(workspace.laps).items()},
            "cross_lap_variability": {k: _r(v, 4) for k, v in variability.items()},
        },
    }
