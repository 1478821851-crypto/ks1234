"""Find occupied game roster cells from repeated card backgrounds.

Brightness projections cannot distinguish a '+' slot from text and can lose
sparse final rows. Background components give tight edges; neighbouring cards
recover cells obscured by avatar frames or dimmed when a player is offline.
No OCR output, member names, screenshot identifiers or fixed counts are used.
"""
import cv2
import numpy as np


def _median(items, index):
    return float(np.median([box[index] for box in items]))


def _width(items):
    return float(np.median([box[2] - box[0] for box in items]))


def _height(items):
    return float(np.median([box[3] - box[1] for box in items]))


def occupied_card_grid(rgb):
    """Return tight occupied cells, or None when no repeated grid is evident.

    Work in the original resolution. Each stacked panel learns its own card
    width and columns, including panels enlarged to two or one columns.
    """
    height, width = rgb.shape[:2]
    brightest = rgb.max(axis=2).astype(np.int16)
    chroma = brightest - rgb.min(axis=2)
    mask = ((chroma > 40) & (brightest > 85)).astype(np.uint8) * 255
    kernel_size = max(3, round(width * .002)) | 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    seeds = []
    for left, top, card_width, card_height, area in stats[1:]:
        if (card_width > width * .07 and card_height > height * .005
                and 1.6 < card_width / card_height < 7
                and area / (card_width * card_height) > .65):
            seeds.append(tuple(map(int, (
                left, top, left + card_width, top + card_height,
            ))))
    # Small member lists and non-roster screenshots retain the legacy detector.
    if len(seeds) < 6:
        return None
    ratios = [(box[2] - box[0]) / (box[3] - box[1]) for box in seeds]
    typical_ratio = float(np.median(ratios))
    # A highlighted avatar may join two vertically adjacent backgrounds.
    seeds = [box for box, ratio in zip(seeds, ratios)
             if .65 * typical_ratio < ratio < 1.4 * typical_ratio]

    rows = []
    for box in sorted(seeds, key=lambda item: (item[1], item[0])):
        card_height = box[3] - box[1]
        row = next((row for row in rows
                    if abs(_median(row, 1) - box[1]) < card_height * .5), None)
        # A message banner can shift a seed's top within the same card row.
        if row is None:
            rows.append([box])
        else:
            row.append(box)

    panels = []
    for row in rows:
        if panels:
            previous = panels[-1][-1]
            separate = (
                _median(row, 1) - _median(previous, 1)
                > 1.35 * max(_height(row), _height(previous))
                or not .85 < _width(row) / _width(previous) < 1.18
            )
        if not panels or separate:
            panels.append([row])
        else:
            panels[-1].append(row)

    boxes = []
    for panel in panels:
        panel_seeds = [box for row in panel for box in row]
        # Keep small enlarged panels once the overall roster is established.
        if len(panel_seeds) < 2:
            continue
        typical_width = _width(panel_seeds)
        columns = []
        for box in sorted(panel_seeds, key=lambda item: item[0]):
            center = (box[0] + box[2]) / 2
            column = next((column for column in columns
                           if abs((_median(column, 0) + _median(column, 2)) / 2
                                  - center) < typical_width * .25), None)
            if column is None:
                columns.append([box])
            else:
                column.append(box)
        bounds = [(round(_median(column, 0)), round(_median(column, 2)))
                  for column in columns]
        for row in panel:
            top, bottom = round(_median(row, 1)), round(_median(row, 3))
            for left, right in bounds:
                # Use background occupancy instead of text ink. This excludes
                # '+' slots and banners while retaining dark offline members.
                occupied = np.mean(
                    (chroma[top:bottom, left:right] > 25)
                    & (brightest[top:bottom, left:right] > 65)
                )
                if occupied > .45:
                    boxes.append((left, top, right, bottom))
    return boxes or None
