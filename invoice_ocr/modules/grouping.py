"""
grouping.py - Group OCR text boxes into structured text blocks.

Groups detected text boxes based on three criteria:
  1. Vertical alignment   — boxes on the same horizontal line (similar Y centre)
  2. Font size            — boxes of similar height (≈ same font size)
  3. Distance             — boxes close enough horizontally to belong together

Pipeline:
  raw OCR boxes  →  row_cluster()  →  font_cluster()  →  merge_lines()
                                                            ↓
                                               List[TextGroup] (structured)

Usage:
    from modules.grouping import TextBoxGrouper

    grouper = TextBoxGrouper()
    groups  = grouper.group(ocr_blocks)          # ocr_blocks from InvoiceOCR.run()

    for g in groups:
        print(g["line_text"], "| font_size:", g["avg_font_size"],
              "| conf:", g["avg_confidence"])

Standalone:
    py modules/grouping.py  <ocr_result.json>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ─── Type alias ───────────────────────────────────────────────────────────────
Box   = Dict[str, Any]        # single OCR box dict from InvoiceOCR.run()
Group = Dict[str, Any]        # structured grouped text block


# ─── Default thresholds ───────────────────────────────────────────────────────
DEFAULT_ROW_Y_OVERLAP    = 0.6    # fraction of box height that must overlap to be "same row"
DEFAULT_FONT_SIZE_TOL    = 0.35   # max relative difference in box heights to share a font group
DEFAULT_H_GAP_FACTOR     = 2.5    # max gap = factor × median box height  (horizontal)
DEFAULT_V_GAP_FACTOR     = 1.2    # max gap = factor × median box height  (vertical, between lines)
DEFAULT_MIN_CONFIDENCE   = 0.0    # keep all boxes by default


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _box_height(b: Box) -> float:
    return max(b["y_max"] - b["y_min"], 1.0)


def _box_width(b: Box) -> float:
    return max(b["x_max"] - b["x_min"], 1.0)


def _y_center(b: Box) -> float:
    return (b["y_min"] + b["y_max"]) / 2.0


def _x_center(b: Box) -> float:
    return (b["x_min"] + b["x_max"]) / 2.0


def _y_overlap_ratio(a: Box, b: Box) -> float:
    """Fraction of the shorter box's height that overlaps vertically."""
    top    = max(a["y_min"], b["y_min"])
    bottom = min(a["y_max"], b["y_max"])
    overlap = max(0.0, bottom - top)
    shorter = min(_box_height(a), _box_height(b))
    return overlap / shorter


def _font_size_similarity(a: Box, b: Box, tol: float) -> bool:
    """True if relative height difference ≤ tol."""
    ha, hb = _box_height(a), _box_height(b)
    return abs(ha - hb) / max(ha, hb) <= tol


def _median_height(boxes: List[Box]) -> float:
    if not boxes:
        return 12.0
    return float(np.median([_box_height(b) for b in boxes]))


# ─── Step 1: Cluster into rows by vertical-alignment ─────────────────────────

def cluster_into_rows(
    boxes: List[Box],
    y_overlap_thresh: float = DEFAULT_ROW_Y_OVERLAP,
) -> List[List[Box]]:
    """
    Group boxes that share the same horizontal text line.

    Two boxes belong to the same row when their vertical overlap ratio
    exceeds `y_overlap_thresh` (default 60 %).

    Returns:
        List of rows; each row is a list of boxes sorted left → right.
    """
    if not boxes:
        return []

    # Sort top → bottom by y_center
    sorted_boxes = sorted(boxes, key=_y_center)

    rows: List[List[Box]] = []
    used = [False] * len(sorted_boxes)

    for i, anchor in enumerate(sorted_boxes):
        if used[i]:
            continue
        row = [anchor]
        used[i] = True
        for j, candidate in enumerate(sorted_boxes[i + 1:], start=i + 1):
            if used[j]:
                continue
            if _y_overlap_ratio(anchor, candidate) >= y_overlap_thresh:
                row.append(candidate)
                used[j] = True
        # Sort row left → right
        rows.append(sorted(row, key=lambda b: b["x_min"]))

    logger.debug(f"cluster_into_rows: {len(boxes)} boxes → {len(rows)} rows")
    return rows


# ─── Step 2: Sub-cluster each row by horizontal distance ─────────────────────

def split_row_by_distance(
    row: List[Box],
    h_gap_factor: float = DEFAULT_H_GAP_FACTOR,
    global_median_h: float = 12.0,
) -> List[List[Box]]:
    """
    Split a single row into sub-groups when there is a large horizontal gap.

    Threshold = h_gap_factor × median_box_height of the row (or global).
    """
    if len(row) <= 1:
        return [row]

    row_median_h = _median_height(row) if row else global_median_h
    threshold    = h_gap_factor * max(row_median_h, global_median_h * 0.5)

    groups: List[List[Box]] = []
    current = [row[0]]

    for prev, cur in zip(row, row[1:]):
        gap = cur["x_min"] - prev["x_max"]
        if gap > threshold:
            groups.append(current)
            current = [cur]
        else:
            current.append(cur)

    groups.append(current)
    return groups


# ─── Step 3: Sub-cluster by font size within each horizontal group ────────────

def split_by_font_size(
    group: List[Box],
    font_tol: float = DEFAULT_FONT_SIZE_TOL,
) -> List[List[Box]]:
    """
    Further split a group if boxes have significantly different font sizes.

    Boxes are sorted by height; a new sub-group starts whenever the relative
    height difference exceeds `font_tol`.
    """
    if len(group) <= 1:
        return [group]

    sorted_g = sorted(group, key=_box_height)
    sub_groups: List[List[Box]] = [[sorted_g[0]]]

    for box in sorted_g[1:]:
        placed = False
        for sg in sub_groups:
            rep = sg[0]  # representative of this size class
            if _font_size_similarity(rep, box, font_tol):
                sg.append(box)
                placed = True
                break
        if not placed:
            sub_groups.append([box])

    # Re-sort each sub-group left → right
    return [sorted(sg, key=lambda b: b["x_min"]) for sg in sub_groups]


# ─── Step 4: Merge nearby rows into multi-line blocks ────────────────────────

def merge_close_rows(
    row_groups: List[List[Box]],
    v_gap_factor: float = DEFAULT_V_GAP_FACTOR,
    global_median_h: float = 12.0,
) -> List[List[List[Box]]]:
    """
    Merge consecutive rows whose vertical gap is small (same paragraph / block).

    Returns:
        List of "line-groups"; each line-group is a list of rows.
    """
    if not row_groups:
        return []

    threshold = v_gap_factor * global_median_h
    merged: List[List[List[Box]]] = [[row_groups[0]]]

    for prev_row, cur_row in zip(row_groups, row_groups[1:]):
        prev_bottom = max(b["y_max"] for b in prev_row)
        cur_top     = min(b["y_min"] for b in cur_row)
        v_gap       = cur_top - prev_bottom

        if v_gap <= threshold:
            merged[-1].append(cur_row)
        else:
            merged.append([cur_row])

    return merged


# ─── Step 5: Build structured TextGroup dicts ─────────────────────────────────

def build_group(rows: List[List[Box]]) -> Group:
    """
    Combine a list of rows into one structured TextGroup dict.

    Fields:
        line_text       (str)   – full text of the group (lines joined by \\n)
        boxes           (list)  – all constituent boxes
        avg_confidence  (float) – mean confidence of all boxes
        avg_font_size   (float) – mean box height (proxy for font size)
        x_min, y_min, x_max, y_max (float) – bounding box of the whole group
        row_count       (int)   – number of text lines
        box_count       (int)   – number of individual text boxes
    """
    all_boxes: List[Box] = [b for row in rows for b in row]
    if not all_boxes:
        return {}

    line_texts = [" ".join(b["text"] for b in row) for row in rows]

    return {
        "line_text":      "\n".join(line_texts),
        "lines":          line_texts,
        "boxes":          all_boxes,
        "avg_confidence": round(
            sum(b["confidence"] for b in all_boxes) / len(all_boxes), 4
        ),
        "avg_font_size":  round(_median_height(all_boxes), 2),
        "x_min":          round(min(b["x_min"] for b in all_boxes), 2),
        "y_min":          round(min(b["y_min"] for b in all_boxes), 2),
        "x_max":          round(max(b["x_max"] for b in all_boxes), 2),
        "y_max":          round(max(b["y_max"] for b in all_boxes), 2),
        "row_count":      len(rows),
        "box_count":      len(all_boxes),
    }


# ─── Main grouper class ───────────────────────────────────────────────────────

class TextBoxGrouper:
    """
    Groups raw OCR text boxes into structured text blocks.

    Grouping criteria (applied in order):
      1. Vertical alignment   — same horizontal line  (Y-overlap ratio)
      2. Horizontal distance  — close boxes on the same line
      3. Font size            — similar bounding-box height
      4. Vertical proximity   — nearby lines form a multi-line block

    Usage:
        grouper = TextBoxGrouper()
        groups  = grouper.group(ocr_blocks)
    """

    def __init__(
        self,
        y_overlap_thresh : float = DEFAULT_ROW_Y_OVERLAP,
        font_size_tol    : float = DEFAULT_FONT_SIZE_TOL,
        h_gap_factor     : float = DEFAULT_H_GAP_FACTOR,
        v_gap_factor     : float = DEFAULT_V_GAP_FACTOR,
        min_confidence   : float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.y_overlap_thresh = y_overlap_thresh
        self.font_size_tol    = font_size_tol
        self.h_gap_factor     = h_gap_factor
        self.v_gap_factor     = v_gap_factor
        self.min_confidence   = min_confidence

    # ── Public API ────────────────────────────────────────────────────────

    def group(self, boxes: List[Box]) -> List[Group]:
        """
        Group raw OCR boxes into structured text blocks.

        Args:
            boxes: List of dicts from InvoiceOCR.run()
                   (must have: text, confidence, x_min, y_min, x_max, y_max)

        Returns:
            List of TextGroup dicts, sorted top-to-bottom then left-to-right.
            Each group has:
                line_text, lines, boxes,
                avg_confidence, avg_font_size,
                x_min, y_min, x_max, y_max,
                row_count, box_count
        """
        # Filter by confidence
        filtered = [b for b in boxes if b.get("confidence", 1.0) >= self.min_confidence]
        if not filtered:
            logger.warning("No boxes after confidence filter.")
            return []

        global_median_h = _median_height(filtered)
        logger.debug(f"Global median font height: {global_median_h:.1f}px")

        # Step 1 — Cluster into rows
        rows = cluster_into_rows(filtered, self.y_overlap_thresh)

        # Step 2 — Split each row by horizontal gaps & font size
        all_sub_rows: List[List[Box]] = []
        for row in rows:
            h_groups = split_row_by_distance(row, self.h_gap_factor, global_median_h)
            for hg in h_groups:
                font_groups = split_by_font_size(hg, self.font_size_tol)
                all_sub_rows.extend(font_groups)

        # Re-sort rows top → bottom
        all_sub_rows.sort(key=lambda r: min(b["y_min"] for b in r))

        # Step 3 — Merge vertically close rows into multi-line blocks
        merged = merge_close_rows(all_sub_rows, self.v_gap_factor, global_median_h)

        # Step 4 — Build structured group dicts
        groups = [build_group(block_rows) for block_rows in merged]
        groups = [g for g in groups if g]  # remove empty

        # Sort final groups top → bottom, left → right
        groups.sort(key=lambda g: (g["y_min"], g["x_min"]))

        logger.info(
            f"TextBoxGrouper: {len(filtered)} boxes → {len(groups)} structured groups"
        )
        return groups

    # ── Convenience filters ───────────────────────────────────────────────

    @staticmethod
    def filter_by_font_size(
        groups: List[Group],
        min_size: float = 0.0,
        max_size: float = float("inf"),
    ) -> List[Group]:
        """Return groups whose avg_font_size falls within [min_size, max_size]."""
        return [g for g in groups if min_size <= g["avg_font_size"] <= max_size]

    @staticmethod
    def filter_by_confidence(
        groups: List[Group],
        threshold: float = 0.7,
    ) -> List[Group]:
        """Return groups with avg_confidence ≥ threshold."""
        return [g for g in groups if g["avg_confidence"] >= threshold]

    @staticmethod
    def get_largest_font_group(groups: List[Group]) -> Group | None:
        """Return the group with the largest average font size (likely a title/header)."""
        return max(groups, key=lambda g: g["avg_font_size"], default=None)

    @staticmethod
    def to_plain_text(groups: List[Group]) -> str:
        """Concatenate all group texts in reading order."""
        return "\n\n".join(g["line_text"] for g in groups if g.get("line_text"))


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: py modules/grouping.py <ocr_result.json>")
        print("  (JSON must be the output of InvoiceOCR.run())")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        ocr_boxes = json.load(f)

    grouper = TextBoxGrouper()
    groups  = grouper.group(ocr_boxes)

    print(f"\n{'='*70}")
    print(f"  Text Box Grouping Results")
    print(f"  Source : {json_path.name}")
    print(f"  Input boxes  : {len(ocr_boxes)}")
    print(f"  Output groups: {len(groups)}")
    print(f"{'='*70}\n")

    for i, g in enumerate(groups, 1):
        print(f"┌── Group {i:02d} "
              f"| rows={g['row_count']} boxes={g['box_count']} "
              f"| font_size={g['avg_font_size']:.1f}px "
              f"| conf={g['avg_confidence']:.4f}")
        print(f"│   BBox: ({g['x_min']}, {g['y_min']}) → ({g['x_max']}, {g['y_max']})")
        for line in g["lines"]:
            print(f"│   {line}")
        print("└" + "─" * 68)

    # Save grouped output as JSON
    out_path = json_path.parent / f"{json_path.stem}_grouped.json"
    # Exclude raw 'boxes' list to keep JSON clean
    slim = [{k: v for k, v in g.items() if k != "boxes"} for g in groups]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"\nGrouped output saved → {out_path}")
