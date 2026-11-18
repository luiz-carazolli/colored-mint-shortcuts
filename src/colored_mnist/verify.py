"""Verification checks, digit-by-color frequency tables, and validation grid rendering."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from colored_mnist.assignments import (
    HeldoutViewRecord,
    TrainDrawRecord,
    build_all_assignments,
)
from colored_mnist.config import DatasetConfig
from colored_mnist.palette import Palette, composite_digit_srgb
from colored_mnist.splits import RawMNISTData, SplitRecord, build_stratified_splits


@dataclass(frozen=True)
class FrequencySummary:
    """Realized vs. expected preferred-color rate for one assignment condition."""

    condition: str
    total: int
    preferred_count: int
    realized_rate: float
    expected_rate: float
    z_score: float | None
    counts_10x10: list[list[int]]


def _binomial_z(k: int, n: int, p: float) -> float | None:
    if p <= 0.0 or p >= 1.0 or n == 0:
        return None
    sd = math.sqrt(n * p * (1.0 - p))
    if sd == 0.0:
        return None
    return (k - n * p) / sd


def _tabulate_counts(labels: list[int], color_ids: list[int]) -> list[list[int]]:
    table = np.zeros((10, 10), dtype=np.int64)
    for y, c in zip(labels, color_ids):
        table[y, c] += 1
    return table.tolist()


def verify_splits(raw: RawMNISTData, splits: list[SplitRecord], config: DatasetConfig) -> dict[str, Any]:
    """Verify split sizes, disjoint original IDs, label consistency, and stratification."""
    by_split: dict[str, list[SplitRecord]] = {"train": [], "val": [], "test": []}
    for rec in splits:
        if rec.split not in by_split:
            raise AssertionError(f"Unexpected split {rec.split!r}")
        by_split[rec.split].append(rec)

    train_ids = {r.original_id for r in by_split["train"]}
    val_ids = {r.original_id for r in by_split["val"]}
    test_ids = {r.original_id for r in by_split["test"]}

    assert len(train_ids) == len(by_split["train"])
    assert len(val_ids) == len(by_split["val"])
    assert len(test_ids) == len(by_split["test"])
    assert train_ids.isdisjoint(val_ids), "Train and val original_ids overlap"
    assert train_ids.isdisjoint(test_ids), "Train and test original_ids overlap"
    assert val_ids.isdisjoint(test_ids), "Val and test original_ids overlap"

    assert len(train_ids) + len(val_ids) == len(raw.train_labels)
    assert len(test_ids) == len(raw.test_labels)

    # Check stratification & raw label agreement
    val_counts = [0] * 10
    train_counts = [0] * 10
    test_counts = [0] * 10
    for rec in splits:
        _, raw_label = raw.get_image_and_label(rec.source_split, rec.source_index)
        assert rec.label == raw_label, f"Label mismatch for {rec.original_id}"
        if rec.split == "train":
            train_counts[rec.label] += 1
        elif rec.split == "val":
            val_counts[rec.label] += 1
        else:
            test_counts[rec.label] += 1

    expected_total_val = int(round(len(raw.train_labels) * config.val_fraction))
    assert len(val_ids) == expected_total_val, (
        f"Total val count {len(val_ids)} != expected {expected_total_val}"
    )

    for d in range(10):
        total_official_train_d = train_counts[d] + val_counts[d]
        exact_val_d = total_official_train_d * config.val_fraction
        assert abs(val_counts[d] - exact_val_d) < 1.0, (
            f"Digit {d} val count {val_counts[d]} deviates >= 1 from stratified target {exact_val_d:.2f}"
        )

    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "disjoint_ids_verified": True,
        "stratified_class_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
    }


def verify_heldout_views(
    raw: RawMNISTData,
    splits: list[SplitRecord],
    heldout: list[HeldoutViewRecord],
    palette: Palette,
) -> dict[str, Any]:
    """Verify 1-to-1 ID pairing, invariant labels/masks, and valid swapped targets."""
    expected_heldout_ids = [r.original_id for r in splits if r.split in ("val", "test")]
    actual_heldout_ids = [r.original_id for r in heldout]
    assert actual_heldout_ids == expected_heldout_ids, "Held-out IDs do not match split manifest 1-to-1"

    for rec in heldout:
        mask_u8, raw_label = raw.get_image_and_label(rec.source_split, rec.source_index)
        assert rec.label == raw_label
        assert rec.aligned_color_id == palette.preferred_color_by_digit[rec.label]
        assert rec.swapped_target_class != rec.label, (
            f"Swapped target equals true label for {rec.original_id}"
        )
        assert rec.swapped_color_id == palette.preferred_color_by_digit[rec.swapped_target_class]

        # Verify foreground mask invariance across rendered views:
        # At background pixels (mask == 0), rendered RGB equals quantized sRGB palette entry;
        # At full-foreground pixels (mask == 255), rendered RGB equals pure white (255, 255, 255).
        # Check on the first 50 items to stay fast while testing every pixel invariant.
        if rec.source_index < 50:
            imgs = [
                composite_digit_srgb(mask_u8, palette.bg_linear_table[c])
                for c in (rec.aligned_color_id, rec.independent_color_id, rec.swapped_color_id)
            ]
            bg_mask = mask_u8 == 0
            fg_mask = mask_u8 == 255
            for img, cid in zip(
                imgs, (rec.aligned_color_id, rec.independent_color_id, rec.swapped_color_id)
            ):
                if np.any(bg_mask):
                    expected_rgb = np.array(palette.colors[cid].srgb_u8, dtype=np.uint8)
                    assert np.all(img[bg_mask] == expected_rgb)
                if np.any(fg_mask):
                    assert np.all(img[fg_mask] == 255)

    return {
        "heldout_total": len(heldout),
        "one_to_one_ids_verified": True,
        "swapped_targets_distinct_verified": True,
        "mask_and_label_invariance_verified": True,
    }


def verify_training_draw_reuse(
    train_by_seed: dict[int, list[TrainDrawRecord]],
    splits: list[SplitRecord],
    config: DatasetConfig,
) -> dict[str, Any]:
    """Verify training draws reuse the same original digits and alternatives across p."""
    expected_train_ids = [r.original_id for r in splits if r.split == "train"]
    p_sorted = sorted(config.p_values)

    for seed, draws in train_by_seed.items():
        assert [d.original_id for d in draws] == expected_train_ids, (
            f"Train IDs for seed {seed} do not match split manifest"
        )
        for d in draws:
            assert d.alt_color_id != d.preferred_color_id
            # Monotonicity under shared draw: once u < p_low, it must stay preferred for all p_high > p_low;
            # whenever u >= p, it must use the exact same alt_color_id.
            seen_preferred = False
            for p in p_sorted:
                c = d.color_for_p(p)
                if d.uniform_draw < p:
                    assert c == d.preferred_color_id
                    seen_preferred = True
                else:
                    assert not seen_preferred
                    assert c == d.alt_color_id

    return {
        "dataset_seeds_verified": list(config.dataset_seeds),
        "shared_draws_across_p_verified": True,
    }


def verify_determinism(
    raw: RawMNISTData,
    palette: Palette,
    config: DatasetConfig,
    splits: list[SplitRecord],
    train_by_seed: dict[int, list[TrainDrawRecord]],
    heldout: list[HeldoutViewRecord],
) -> dict[str, Any]:
    """Regenerate splits, assignments, and sample pixels from scratch and assert equality."""
    splits_2 = build_stratified_splits(raw, config)
    train_2, heldout_2 = build_all_assignments(splits_2, palette, config)

    assert splits_2 == splits
    assert train_2 == train_by_seed
    assert heldout_2 == heldout

    # Check pixel-exact determinism on first 20 held-out items
    for r in heldout[:20]:
        mask_u8, _ = raw.get_image_and_label(r.source_split, r.source_index)
        img_a = composite_digit_srgb(mask_u8, palette.bg_linear_table[r.swapped_color_id])
        img_b = composite_digit_srgb(mask_u8, palette.bg_linear_table[r.swapped_color_id])
        assert np.array_equal(img_a, img_b)

    return {"deterministic_regeneration_verified": True}


def compute_frequency_summaries(
    train_by_seed: dict[int, list[TrainDrawRecord]],
    heldout: list[HeldoutViewRecord],
    palette: Palette,
    config: DatasetConfig,
) -> list[FrequencySummary]:
    """Tabulate digit-by-color counts and compare realized preferred rates against targets."""
    summaries: list[FrequencySummary] = []

    for seed in config.dataset_seeds:
        draws = train_by_seed[seed]
        labels = [d.label for d in draws]
        for p in config.p_values:
            colors = [d.color_for_p(p) for d in draws]
            pref_count = sum(1 for d in draws if d.color_for_p(p) == d.preferred_color_id)
            n = len(draws)
            rate = pref_count / n
            z = _binomial_z(pref_count, n, p)
            if z is not None:
                assert abs(z) < 4.5, (
                    f"Realized preferred rate {rate:.4f} deviates unexpectedly from p={p:.2f} (z={z:.2f})"
                )
            elif p == 1.0:
                assert pref_count == n
            summaries.append(
                FrequencySummary(
                    condition=f"train_seed_{seed}_p_{p:.2f}",
                    total=n,
                    preferred_count=pref_count,
                    realized_rate=rate,
                    expected_rate=p,
                    z_score=z,
                    counts_10x10=_tabulate_counts(labels, colors),
                )
            )

    for split_name in ("val", "test"):
        subset = [r for r in heldout if r.split == split_name]
        labels = [r.label for r in subset]
        n = len(subset)
        for view, expected_p in (("aligned", 1.0), ("independent", 0.10), ("swapped", 0.0)):
            colors = [r.color_for_view(view) for r in subset]
            pref_count = sum(
                1 for r in subset if r.color_for_view(view) == palette.preferred_color_by_digit[r.label]
            )
            rate = pref_count / n
            z = _binomial_z(pref_count, n, expected_p)
            if z is not None:
                assert abs(z) < 4.5, (
                    f"{split_name}/{view} realized preferred rate {rate:.4f} deviates from {expected_p} (z={z:.2f})"
                )
            elif expected_p == 1.0:
                assert pref_count == n
            elif expected_p == 0.0:
                assert pref_count == 0
            summaries.append(
                FrequencySummary(
                    condition=f"{split_name}_{view}",
                    total=n,
                    preferred_count=pref_count,
                    realized_rate=rate,
                    expected_rate=expected_p,
                    z_score=z,
                    counts_10x10=_tabulate_counts(labels, colors),
                )
            )

    return summaries


def save_digit_color_counts_csv(summaries: list[FrequencySummary], path: Path) -> None:
    """Save 10x10 digit-by-color count tables for every condition to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "digit", *[f"color_{c}" for c in range(10)], "row_total", "preferred_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            for digit in range(10):
                row_counts = s.counts_10x10[digit]
                row_total = sum(row_counts)
                row: dict[str, object] = {
                    "condition": s.condition,
                    "digit": digit,
                    "row_total": row_total,
                    "preferred_rate": f"{(row_counts[digit] / row_total) if row_total else 0.0:.4f}",
                }
                for c in range(10):
                    row[f"color_{c}"] = row_counts[c]
                writer.writerow(row)


def render_validation_sample_grid(
    raw: RawMNISTData,
    splits: list[SplitRecord],
    palette: Palette,
    output_path: Path,
    scale: int = 3,
    padding: int = 8,
) -> Path:
    """Render an annotated 10x10 grid of validation digits (rows) across all 10 colors (cols)."""
    val_by_digit: dict[int, SplitRecord] = {}
    for rec in splits:
        if rec.split == "val" and rec.label not in val_by_digit:
            val_by_digit[rec.label] = rec
            if len(val_by_digit) == 10:
                break

    tile = 28 * scale  # 84 px
    left_margin = 92
    top_margin = 62
    width = left_margin + 10 * (tile + padding) + padding
    height = top_margin + 10 * (tile + padding) + padding

    canvas = Image.new("RGB", (width, height), color=(248, 249, 250))
    draw = ImageDraw.Draw(canvas)

    # Column headers: Color ID, hex, contrast ratio
    for c_id, color_spec in enumerate(palette.colors):
        x0 = left_margin + c_id * (tile + padding)
        draw.text((x0 + 6, 8), f"Color {c_id}", fill=(25, 25, 25))
        draw.text((x0 + 6, 22), color_spec.hex_code, fill=(75, 75, 75))
        draw.text((x0 + 6, 36), f"CR {color_spec.contrast_vs_white:.2f}", fill=(75, 75, 75))

    # Rows: one validation digit per class 0..9 rendered on all 10 background colors
    for digit in range(10):
        rec = val_by_digit[digit]
        mask_u8, _ = raw.get_image_and_label(rec.source_split, rec.source_index)
        y0 = top_margin + digit * (tile + padding)
        draw.text((10, y0 + tile // 2 - 12), f"Digit {digit}", fill=(25, 25, 25))
        draw.text((10, y0 + tile // 2 + 2), f"({rec.source_index:05d})", fill=(100, 100, 100))

        pref_color = palette.preferred_color_by_digit[digit]
        swapped_target = palette.swapped_target_by_digit[digit]
        swapped_color = palette.preferred_color_by_digit[swapped_target]

        for c_id in range(10):
            x0 = left_margin + c_id * (tile + padding)
            img_u8 = composite_digit_srgb(mask_u8, palette.bg_linear_table[c_id])
            pil_tile = Image.fromarray(img_u8, mode="RGB").resize(
                (tile, tile), resample=Image.Resampling.NEAREST
            )
            canvas.paste(pil_tile, (x0, y0))

            # Subtle border markers: white/dark frame on preferred color, dashed-like amber on swapped
            if c_id == pref_color:
                draw.rectangle([x0 - 2, y0 - 2, x0 + tile + 1, y0 + tile + 1], outline=(20, 20, 20), width=2)
            elif c_id == swapped_color:
                draw.rectangle([x0 - 2, y0 - 2, x0 + tile + 1, y0 + tile + 1], outline=(210, 120, 20), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def run_all_verifications(
    raw: RawMNISTData,
    splits: list[SplitRecord],
    train_by_seed: dict[int, list[TrainDrawRecord]],
    heldout: list[HeldoutViewRecord],
    palette: Palette,
    config: DatasetConfig,
) -> tuple[dict[str, Any], list[FrequencySummary]]:
    """Run all Step 1 verification checks and return structured report + count summaries."""
    split_report = verify_splits(raw, splits, config)
    heldout_report = verify_heldout_views(raw, splits, heldout, palette)
    train_report = verify_training_draw_reuse(train_by_seed, splits, config)
    determinism_report = verify_determinism(raw, palette, config, splits, train_by_seed, heldout)
    freq_summaries = compute_frequency_summaries(train_by_seed, heldout, palette, config)

    contrasts = [c.contrast_vs_white for c in palette.colors]
    luminances = [c.relative_luminance for c in palette.colors]

    report: dict[str, Any] = {
        "status": "PASSED",
        "splits": split_report,
        "heldout_views": heldout_report,
        "training_draws": train_report,
        "determinism": determinism_report,
        "palette_summary": {
            "num_colors": len(palette.colors),
            "min_contrast_vs_white": round(min(contrasts), 4),
            "max_contrast_vs_white": round(max(contrasts), 4),
            "min_relative_luminance": round(min(luminances), 6),
            "max_relative_luminance": round(max(luminances), 6),
        },
        "frequency_checks": [
            {
                "condition": s.condition,
                "total": s.total,
                "preferred_count": s.preferred_count,
                "realized_rate": round(s.realized_rate, 6),
                "expected_rate": s.expected_rate,
                "z_score": round(s.z_score, 3) if s.z_score is not None else None,
            }
            for s in freq_summaries
        ],
    }
    return report, freq_summaries
