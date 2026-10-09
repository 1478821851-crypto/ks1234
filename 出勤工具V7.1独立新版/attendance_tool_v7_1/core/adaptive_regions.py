"""OCR-independent local segmentation, without a colour palette or fixed grid."""
import numpy as np


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def bridge(mask,gap):
    result=mask.copy()
    for a,b in runs(~mask):
        if a>0 and b<len(mask) and b-a<=gap:
            result[a:b]=True
    return result


def otsu(values):
    hist=np.bincount(np.clip(values,0,255).astype(np.uint8).ravel(),minlength=256).astype(float)
    weight=np.cumsum(hist); total=weight[-1]
    mean=np.cumsum(hist*np.arange(256))
    score=(mean[-1]*weight-mean*total)**2/np.maximum(weight*(total-weight),1)
    return int(np.argmax(score))


def grid_boxes(image,roi,rows,columns):
    h,w=image.shape[:2]; l,t,r,b=roi
    xs=np.linspace(round(w*l/100),round(w*r/100),columns+1).round().astype(int)
    ys=np.linspace(round(h*t/100),round(h*b/100),rows+1).round().astype(int)
    return [dict(box=(int(xs[j]),int(ys[i]),int(xs[j+1]),int(ys[i+1])),reliable=True,method="人工设置切图")
            for i in range(rows) for j in range(columns) if xs[j+1]>xs[j] and ys[i+1]>ys[i]]


def _detect_grid_sections(rgb):
    """Detect horizontal breaks between stacked grid sections.

    Game screenshots may contain multiple grid panels stacked vertically,
    separated by a wide dark band (the UI separator). Returns a list of
    (top, bottom) row ranges for each section.
    """
    import cv2
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=1)
    median_brightness = float(np.median(profile))

    # Find long runs of very-dark rows (the separator band)
    dark_mask = profile < median_brightness * 0.55
    dark_runs_list = runs(dark_mask)
    # Separators are interior dark bands clearly wider than the row gutters
    # around them. Gutters scale with card size, so compare each candidate
    # against the narrow dark runs in its neighbourhood, not a global fraction.
    inner = [(a, b) for a, b in dark_runs_list
             if a > h * 0.015 and b < h * 0.985]
    narrow_cap = max(12, int(h * 0.008))
    narrow_runs = [(a, b) for a, b in inner if b - a <= narrow_cap]
    narrow = [(int((a + b) / 2), b - a) for a, b in narrow_runs]
    all_narrow = [ww for _, ww in narrow]
    win = max(300, int(h * 0.15))
    separators = []
    for a, b in inner:
        wdt = b - a
        center = (a + b) // 2
        neigh = [ww for cc, ww in narrow if abs(cc - center) <= win]
        local = float(np.median(neigh)) if neigh else (
            float(np.median(all_narrow)) if all_narrow else 0.0)
        # A separator must be clearly wider than the *local* row gutters.
        # Do not hard-skip runs <= narrow_cap: that cap scales with whole
        # image height, while gutters do not, so a short separator inside
        # a tall composite screenshot (e.g. a 26px divider among 11px
        # gutters) would be mislabelled as a gutter and never re-checked.
        if local > 0 and wdt > local * 2.2 and wdt > local + 6:
            separators.append((a, b))

    # Stacked grids may use different column counts (e.g. a 5-column panel
    # pasted above a 4-column panel), divided only by a semi-transparent UI
    # band whose brightness (60-70 here) sits just above the global dark
    # threshold, so the wide-band rule above misses the split. The row rhythm
    # still breaks cleanly: one gap between neighbouring narrow row gutters
    # is far larger than the median pitch. Anchor the separator on the darkest
    # run inside that gap (the band core) rather than on absolute brightness.
    if len(narrow_runs) >= 4:
        centers = [int((a + b) / 2) for a, b in narrow_runs]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        med_gap = float(np.median(gaps))
        for i, gap in enumerate(gaps):
            if not (gap > med_gap * 1.8 and gap > med_gap + 20
                    and gap < med_gap * 3.5):
                continue
            ga, gb = narrow_runs[i][1], narrow_runs[i + 1][0]
            core_runs = [(a, b) for a, b in inner
                         if ga < a and b < gb and b - a >= 6]
            if core_runs:
                ca, cb = min(core_runs,
                             key=lambda ab: float(profile[ab[0]:ab[1]].mean()))
            else:
                ystar = int(np.argmin(profile[ga:gb])) + ga
                ca, cb = max(ga, ystar - 3), min(gb, ystar + 4)
            # The core must sit inside the gap (not touch either gutter) and
            # be clearly darker than the card rows flanking it.
            if ca - ga < 20 or gb - cb < 20:
                continue
            flank = np.concatenate([
                profile[max(0, ca - 80):ca],
                profile[cb:min(h, cb + 80)]])
            if flank.size and float(profile[ca:cb].mean()) < 0.85 * float(flank.mean()):
                separators.append((ca, cb))

    if not separators:
        return [(0, h)]

    # Merge overlapping separators emitted by either rule, then sort.
    separators.sort()
    merged_seps = []
    for a, b in separators:
        if merged_seps and a <= merged_seps[-1][1]:
            merged_seps[-1] = (merged_seps[-1][0], max(merged_seps[-1][1], b))
        else:
            merged_seps.append((a, b))

    sections = []
    prev_bottom = 0
    for a, b in merged_seps:
        if a - prev_bottom > h * 0.03:
            sections.append((prev_bottom, a))
        prev_bottom = b
    if prev_bottom < h:
        sections.append((prev_bottom, h))
    return [(t, b) for t, b in sections if b - t > h * 0.04]


def _content_area(section_rgb):
    """Detect and return (top, bottom) of the content area, excluding UI bars."""
    import cv2
    h = section_rgb.shape[0]
    if h <= 50:
        return (0, h)
    gray = cv2.cvtColor(section_rgb, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=1)
    p30 = float(np.percentile(profile, 30))
    p40 = float(np.percentile(profile, 40))

    content_top = 0
    if profile[0] < p30:
        for y in range(min(h, 200)):
            if profile[y] > p40:
                content_top = y
                break

    content_bottom = h
    if profile[-1] < p30:
        for y in range(h - 1, max(0, h - 200), -1):
            if profile[y] > p40:
                content_bottom = y
                break

    return (content_top, content_bottom)


def _find_columns_in_section(section_rgb):
    """Find vertical column boundaries within one grid section.

    Cards have bright colored content; gutters between columns are narrow
    dark bands. We detect columns by finding these dark vertical strips
    in the per-column brightness profile across the content area height.
    """
    h, w = section_rgb.shape[:2]
    ct, cb = _content_area(section_rgb)
    content = section_rgb[ct:cb, :]
    brightness = content.mean(2).mean(0)

    kernel_size = max(1, min(3, w // 500))
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(brightness, kernel, mode='same')
    else:
        smoothed = brightness

    # Gutters are darker than their local neighbourhood. A global percentile
    # fails when gutters cover only a small fraction of the width: the
    # threshold then eats into dark cards and merges gutters into wide runs.
    win = max(40, w // 12)
    pad = np.pad(smoothed, win, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * win + 1)
    base = np.median(windows, axis=1)
    dark_mask = smoothed < base * 0.70
    dark_runs_list = runs(dark_mask)
    gutter_centers = []
    edge_margin = int(w * 0.03)
    for a, b in dark_runs_list:
        width = b - a
        if 2 <= width <= max(12, int(w * 0.05)):
            center = int((a + b) / 2)
            if center > edge_margin and center < w - edge_margin:
                gutter_centers.append(center)

    cuts = [0] + sorted(set(gutter_centers)) + [w]
    merged = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] > w * 0.04:
            merged.append(c)
    if merged[-1] != w:
        merged.append(w)

    if len(merged) <= 2:
        return [(0, w)]

    cols = list(zip(merged, merged[1:]))

    # Pre-filter: remove very narrow columns that are clearly artifacts
    max_col_w = max(cr - cl for cl, cr in cols)
    cols = [(cl, cr) for cl, cr in cols if cr - cl >= max_col_w * 0.30]

    if len(cols) < 2:
        return [(0, w)]

    # Merge abnormally narrow columns (likely split columns from border artifacts)
    widths = [cr - cl for cl, cr in cols]
    median_w = float(np.median(widths))
    final = []
    skip = set()
    i = 0
    while i < len(cols):
        if i in skip:
            i += 1
            continue
        cl, cr = cols[i]
        cw = cr - cl
        
        # Check if this column is very narrow (< 45% of median)
        if cw < median_w * 0.45:
            if final:
                # Merge with previous column
                fl, fr = final[-1]
                final[-1] = (fl, cr)
            else:
                # Merge with next column
                if i + 1 < len(cols) and i + 1 not in skip:
                    nc = cols[i + 1][1]
                    final.append((cl, nc))
                    skip.add(i + 1)
                else:
                    final.append((cl, cr))
        # Check if this and next column are both half-width (50-75% of median)
        elif cw < median_w * 0.75 and i + 1 < len(cols) and i + 1 not in skip:
            next_cl, next_cr = cols[i + 1]
            next_cw = next_cr - next_cl
            if next_cw < median_w * 0.75 and (cw + next_cw) > median_w * 0.85:
                # Merge these two half-width columns
                final.append((cl, next_cr))
                skip.add(i + 1)
            else:
                final.append((cl, cr))
        else:
            final.append((cl, cr))
        i += 1
    return final


def _find_rows_in_section(section_rgb, num_cols_hint=None):
    """Find horizontal row boundaries within one grid section.

    Cards are separated by thin dark gutters. We detect rows by finding
    these dark gaps in the vertical brightness profile. The top/bottom
    UI bars (if present) are detected and excluded first.
    """
    import cv2
    h, w = section_rgb.shape[:2]
    gray = cv2.cvtColor(section_rgb, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=1)

    content_top, content_bottom = _content_area(section_rgb)

    content_profile = profile[content_top:content_bottom]
    if len(content_profile) < 10:
        return [(0, h)]
    median_b = float(np.median(content_profile))
    content_h = content_bottom - content_top

    # Try gutter-based detection. Gutters are darker than their local
    # neighbourhood: a global threshold merges gutters into rows of dark
    # cards, losing them to the width cap. Use a slightly relaxed threshold
    # (0.75) so shallow separators next to UI bars are still caught.
    win = max(24, content_h // 30)
    pad = np.pad(content_profile, win, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * win + 1)
    base = np.median(windows, axis=1)
    dark_mask = np.zeros(h, dtype=bool)
    dark_mask[content_top:content_bottom] = content_profile < base * 0.75
    dark_runs_list = runs(dark_mask)
    max_gutter_h = max(10, int(content_h * 0.045))
    gutter_bands = []
    for a, b in dark_runs_list:
        if not (2 <= b - a <= max_gutter_h
                and a > content_top + 2 and b < content_bottom - 2):
            continue
        # Reject shallow dips inside dark card rows: a real gutter is much
        # darker than the bands on both sides of it.
        flank = max(4, b - a)
        outside = np.concatenate([profile[max(0, a - flank):a], profile[b:min(h, b + flank)]])
        if outside.size and profile[a:b].mean() < 0.7 * outside.mean():
            gutter_bands.append((a, b))

    if len(gutter_bands) >= 1:
        # Check for large gaps between consecutive gutters (indicates stacked grids)
        gutter_positions = [int((a + b) / 2) for a, b in gutter_bands]
        gutter_gaps = [gutter_positions[i + 1] - gutter_positions[i]
                       for i in range(len(gutter_positions) - 1)]
        median_gg = float(np.median(gutter_gaps)) if gutter_gaps else 0

        split_idx = -1
        if median_gg > 0:
            for i, g in enumerate(gutter_gaps):
                if g > median_gg * 1.8 and g > median_gg + 20:
                    split_idx = i
                    break

        if split_idx >= 0:
            # Stacked grids inside one section: recurse on both halves so the
            # lower grid's rows are not swallowed into one oversized row.
            mid = (gutter_bands[split_idx][1] + gutter_bands[split_idx + 1][0]) // 2
            if mid - content_top >= content_h * 0.12 and content_bottom - mid >= content_h * 0.12:
                top_rows = _find_rows_in_section(section_rgb[:mid], num_cols_hint)
                bot_rows = _find_rows_in_section(section_rgb[mid:], num_cols_hint)
                rows = list(top_rows) + [(t + mid, b + mid) for t, b in bot_rows]
                if len(rows) >= 2:
                    return rows
            # Split gutters into two groups and use the larger one
            top_gutters = gutter_bands[:split_idx + 1]
            bottom_gutters = gutter_bands[split_idx + 1:]
            gutter_bands = top_gutters if len(top_gutters) >= len(bottom_gutters) else bottom_gutters

        # Build rows from (possibly split) gutters
        rows = []
        min_row_h = content_h * 0.05
        first_gutter_top = gutter_bands[0][0]
        if first_gutter_top - content_top > min_row_h:
            rows.append((content_top, first_gutter_top))
        for i in range(len(gutter_bands) - 1):
            row_top = gutter_bands[i][1]
            row_bottom = gutter_bands[i + 1][0]
            row_h = row_bottom - row_top
            if row_h > min_row_h:
                rows.append((row_top, row_bottom))
        last_gutter_bottom = gutter_bands[-1][1]
        remaining = content_bottom - last_gutter_bottom
        if remaining >= min_row_h:
            rows.append((last_gutter_bottom, content_bottom))

        # Split oversized rows that swallowed a UI bar (title above first
        # card, or button bar below last card). A row much taller than the
        # median means the gutter next to the UI bar was too shallow to be
        # detected; look inside it for dark seams with a relaxed threshold.
        if len(rows) >= 2:
            median_rh = float(np.median([rb - rt for rt, rb in rows]))
            split_rows = []
            for rt, rb in rows:
                rh = rb - rt
                if rh > median_rh * 1.5 and rh > median_rh + 20:
                    sub_profile = profile[rt:rb]
                    sub_base = float(np.median(sub_profile))
                    sub_dark = sub_profile < sub_base * 0.80
                    sub_seams = [(a + rt, b + rt) for a, b in runs(sub_dark)
                                 if 3 <= b - a <= max(10, int(rh * 0.10))]
                    valid_seams = []
                    for sa, sb_ in sub_seams:
                        flank = max(4, sb_ - sa)
                        outside = np.concatenate([
                            profile[max(0, sa - flank):sa],
                            profile[sb_:min(h, sb_ + flank)]])
                        if outside.size and profile[sa:sb_].mean() < 0.75 * outside.mean():
                            valid_seams.append((sa, sb_))
                    if valid_seams:
                        prev = rt
                        for sa, sb_ in valid_seams:
                            if sa - prev > median_rh * 0.6:
                                split_rows.append((prev, sa))
                            prev = sb_
                        if rb - prev > median_rh * 0.6:
                            split_rows.append((prev, rb))
                        # Drop slivers that are far too short to be a card
                        # (e.g. a title band sliced by its own text gaps).
                        split_rows = [(a, b) for a, b in split_rows
                                      if b - a >= median_rh * 0.45]
                        if not split_rows:
                            split_rows = [(rt, rb)]
                    elif rt < content_top + content_h * 0.2:
                        # Top oversized row: title sits above the card, keep
                        # the bottom part (the card).
                        split_rows.append((rb - int(median_rh), rb))
                    elif rb > content_bottom - content_h * 0.2:
                        # Bottom oversized row: button bar sits below the
                        # card, keep the top part.
                        split_rows.append((rt, rt + int(median_rh)))
                    else:
                        split_rows.append((rt, rb))
                else:
                    split_rows.append((rt, rb))
            rows = split_rows

        if len(rows) >= 2:
            return rows

    # Fallback: autocorrelation to detect row periodicity
    centered = content_profile - median_b
    n = len(centered)
    max_lag = min(n // 2, int(n * 0.30))
    period = None
    if max_lag >= 5:
        autocorr = np.array([np.sum(centered[:n - lag] * centered[lag:]) for lag in range(max_lag)])
        autocorr /= np.maximum(np.sum(centered ** 2), 1)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] and autocorr[i] > 0.08:
                peaks.append((i, autocorr[i]))
        if peaks:
            peaks.sort(key=lambda x: -x[1])
            best_period = None
            best_score = -1
            for p, corr in peaks:
                num_rows = content_h / p
                if 3 <= num_rows <= 12:
                    fundamental = p
                    for mult in [2, 3, 4]:
                        candidate = p * mult
                        if candidate < len(autocorr):
                            for pp, cc in peaks:
                                if abs(pp - candidate) < p * 0.15 and cc > corr * 0.5:
                                    fundamental = pp
                                    break
                            if fundamental != p:
                                break
                    num_rows_fund = content_h / fundamental
                    if 3 <= num_rows_fund <= 10:
                        score = corr * (2.0 if 4 <= num_rows_fund <= 7 else 1.0)
                        if score > best_score:
                            best_score = score
                            best_period = fundamental
            if best_period is not None:
                period = best_period

    if period is not None:
        rows = []
        y = content_top
        while y < content_bottom:
            next_y = min(y + int(period), content_bottom)
            if next_y - y > content_h * 0.06:
                rows.append((y, next_y))
            y = next_y
        if rows:
            # A periodicity peak on a single large card is noise: every split
            # must sit on a real dark dip, otherwise keep the section whole.
            win = max(2, int(period) // 8)
            validated = 0
            for rt, _ in rows[1:]:
                left = profile[max(0, rt - 4 * win):max(0, rt - win)]
                right = profile[min(h, rt + win):min(h, rt + 4 * win)]
                mid = profile[max(0, rt - win):min(h, rt + win)]
                if left.size and right.size and mid.size and \
                        mid.mean() < 0.92 * min(left.mean(), right.mean()):
                    validated += 1
            if validated * 2 < len(rows) - 1:
                return [(content_top, content_bottom)]
            # Merge rows that are significantly shorter than median (sub-period artifact)
            row_heights = [rb - rt for rt, rb in rows]
            median_rh = float(np.median(row_heights))
            if median_rh > 0 and len(rows) >= 4:
                merged = []
                skip = set()
                for i in range(len(rows)):
                    if i in skip:
                        continue
                    rt, rb = rows[i]
                    rh = rb - rt
                    if rh < median_rh * 0.6 and i + 1 < len(rows) and (i + 1) not in skip:
                        # Merge with next row
                        nrt, nrb = rows[i + 1]
                        merged.append((rt, nrb))
                        skip.add(i + 1)
                    else:
                        merged.append((rt, rb))
                rows = merged
            return rows

    return [(content_top, content_bottom)]


def game_boxes(rgb):
    """Detect member-card boxes in game screenshots.

    Repeated background components locate occupied cells independently in
    each stacked panel, including sparse rows and changing column counts.
    When no repeated roster is evident, retain the brightness-profile and
    horizontal-strip detector for small lists and non-grid screenshots.
    """
    from attendance_tool_v7_1.core.game_grid import occupied_card_grid
    grid = occupied_card_grid(rgb)
    if grid is not None:
        return grid

    import cv2
    h, w = rgb.shape[:2]

    # Step 1: detect grid sections (handles stacked panels)
    sections = _detect_grid_sections(rgb)

    all_boxes = []
    prev_col_ranges = None
    prev_sw = 0
    for sec_top, sec_bottom in sections:
        section = rgb[sec_top:sec_bottom, :]
        sh, sw = section.shape[:2]

        # Step 2: find column boundaries
        col_ranges = _find_columns_in_section(section)

        # A sparsely-filled grid (e.g. the last panel of a team roster with
        # only a few cards and many empty "+" slots) has no detectable
        # vertical gutters: the empty slots are as dark as the gutters.
        # Reuse the previous section's column layout when widths match.
        if (len(col_ranges) == 1 and col_ranges[0][1] - col_ranges[0][0] >= sw * 0.95
                and prev_col_ranges is not None and prev_sw == sw
                and len(prev_col_ranges) >= 2):
            row_ranges = _find_rows_in_section(section, num_cols_hint=len(prev_col_ranges))
            # Only adopt the hint when rows actually contain multiple cards:
            # check whether at least one row shows a brightness dip at a
            # hinted gutter position.
            hinted = False
            gray_sec = cv2.cvtColor(section, cv2.COLOR_RGB2GRAY)
            for rt, rb in row_ranges[:3]:
                band = gray_sec[rt:rb, :].mean(0)
                for cl, cr in prev_col_ranges[1:-1]:
                    if cl < 5 or cl >= sw - 5:
                        continue
                    win = max(3, int(sw * 0.01))
                    mid = band[max(0, cl - win):cl + win].mean()
                    side_l = band[max(0, cl - 4 * win):cl - win].mean()
                    side_r = band[cl + win:min(sw, cl + 4 * win)].mean()
                    if mid < 0.92 * min(side_l, side_r):
                        hinted = True
                        break
                if hinted:
                    break
            if hinted:
                col_ranges = prev_col_ranges

        # If only 1 column spanning full width, cards are stacked full-width
        # rows (member list or one oversized card), not a 2-D grid.
        if len(col_ranges) == 1 and col_ranges[0][1] - col_ranges[0][0] >= sw * 0.95:
            row_ranges = _find_rows_in_section(section, num_cols_hint=1)
            for rt, rb in row_ranges:
                box_w = sw
                box_h = rb - rt
                if box_w < sw * 0.04 or box_h < sh * 0.03:
                    continue
                all_boxes.append((0, rt + sec_top, sw, rb + sec_top))
            continue

        # Step 3: find row boundaries
        row_ranges = _find_rows_in_section(section, num_cols_hint=len(col_ranges))

        # Step 3.5: merge sub-rows if rows are much shorter than columns
        # Cards sometimes have internal dark lines that get detected as row gutters,
        # splitting each card into 2 half-height sub-rows. Merge them back.
        # Only trigger when there are clearly too many rows (10+).
        if len(col_ranges) >= 2 and len(row_ranges) >= 10:
            median_col_w = float(np.median([cr - cl for cl, cr in col_ranges]))
            median_row_h = float(np.median([rb - rt for rt, rb in row_ranges]))
            if median_row_h < median_col_w * 0.45:
                merged_rows = []
                for i in range(0, len(row_ranges) - 1, 2):
                    rt = row_ranges[i][0]
                    rb = row_ranges[i + 1][1]
                    merged_rows.append((rt, rb))
                if len(row_ranges) % 2 == 1:
                    merged_rows.append(row_ranges[-1])
                row_ranges = merged_rows

        # Step 4: combine row x column into individual card boxes
        median_row_h = float(np.median([rb - rt for rt, rb in row_ranges])) \
            if row_ranges else 0.0
        median_col_w = sw / float(max(1, len(col_ranges)))
        for rt, rb in row_ranges:
            cols_for_row = col_ranges
            if len(col_ranges) > 1 and median_row_h > 0 and rb - rt > median_row_h * 1.4:
                # A section can mix grids of different column counts; an
                # oversized row band carries its own gutters, so trust them
                # when they disagree with the section-wide column layout.
                row_cols = _find_columns_in_section(section[rt:rb, :])
                if (len(row_cols) < len(col_ranges)
                        and min(cr - cl for cl, cr in row_cols) >= median_col_w * 1.4):
                    cols_for_row = row_cols
            for cl, cr in cols_for_row:
                box_w = cr - cl
                box_h = rb - rt
                # Skip boxes that are too small (likely artifacts)
                if box_w < sw * 0.04 or box_h < sh * 0.03:
                    continue
                all_boxes.append((cl, rt + sec_top, cr, rb + sec_top))

        # Remember this section's column layout for the next (possibly
        # sparsely-filled) section.
        if len(col_ranges) >= 2:
            prev_col_ranges = col_ranges
            prev_sw = sw

    # Verify each box has text content
    return _verify_game_boxes(rgb, all_boxes)


def _detect_horizontal_strips(section, y_offset=0):
    """Fallback: detect full-width horizontal card strips.

    Used when column detection finds only one wide column, indicating the
    screenshot uses horizontal strips rather than a 2-D grid.
    """
    import cv2
    h, w = section.shape[:2]
    gray = cv2.cvtColor(section, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=1)

    def runs_local(mask):
        edges = np.diff(np.r_[False, mask, False].astype(int))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    bright_threshold = float(np.percentile(profile, 90))
    bright_mask = profile > bright_threshold
    bright_runs_list = runs_local(bright_mask)
    header_bands = [(a, b) for a, b in bright_runs_list if 3 <= b - a <= 30]

    min_card_h = max(10, h // 60)
    max_card_h = max(min_card_h + 5, h // 6)

    cards = []
    for i in range(len(header_bands) - 1):
        card_top = header_bands[i][1]
        card_bottom = header_bands[i + 1][0]
        card_h = card_bottom - card_top
        if min_card_h < card_h < max_card_h:
            cards.append((0, card_top + y_offset, w, card_bottom + y_offset))

    return cards


def _verify_game_boxes(rgb, boxes):
    """Filter out boxes that don't contain meaningful text content."""
    import cv2
    result = []
    for x1, y1, x2, y2 in boxes:
        patch = rgb[y1:y2, x1:x2]
        hh, ww = patch.shape[:2]
        if ww == 0 or hh == 0:
            continue
        text = patch[:, int(ww * 0.2):int(ww * 0.8)].max(2).astype(float)
        if text.size == 0:
            continue
        background = cv2.GaussianBlur(text, (0, 0), max(1, ww * 0.025))
        threshold = max(12, float(np.percentile(text, 98) - np.percentile(text, 50)) * 0.25)
        ink = ((text - background) > threshold).mean()
        if ink > 0.03:
            result.append((x1, y1, x2, y2))
    return result


def yy_columns(rgb):
    h,w=rgb.shape[:2]
    difference=np.max(np.abs(np.diff(rgb.astype(np.int16),axis=1)),2)
    live=np.zeros(w-1,int); longest=live.copy()
    for row in difference:
        live=np.where(row>5,live+1,0)
        longest=np.maximum(longest,live)
    candidates=np.flatnonzero(longest>h*.17)
    groups=np.split(candidates,np.flatnonzero(np.diff(candidates)>max(2,w*.03))+1)
    ink=(rgb.mean(2)<190)|((rgb.max(2).astype(int)-rgb.min(2))>40)
    profile=ink.mean(0)
    cuts=[0]
    for group in groups:
        if not len(group):
            continue
        x=int(group[np.argmax(longest[group])])+1
        margin=max(3,int(w*.012))
        blank=min(profile[max(0,x-margin):max(1,x-1)].mean(),
                  profile[min(w-1,x+1):min(w,x+margin)].mean())
        if blank>.12:
            continue
        if .06*w<x<.94*w and x-cuts[-1]>w*.10:
            cuts.append(x)
    cuts.append(w)
    # Seamless pasted panels can share a background. Add only broad full-height
    # blank gutters well inside a pane; gaps between an icon and its name are
    # shorter and are excluded by relative width and interior-margin checks.
    extra=[]
    for left,right in zip(cuts,cuts[1:]):
        width=right-left
        for a,b in runs(ink[:,left:right].mean(0)<.025):
            if b-a>width*.03 and a>width*.18 and b<width*.75:
                extra.append(left+int(a))
    cuts=sorted(set(cuts+extra))
    return list(zip(cuts,cuts[1:]))


def yy_boxes(rgb):
    from attendance_tool_v7_1.core.yy_rows import member_panels
    rows = member_panels(rgb, yy_columns(rgb))
    if rows is not None:
        return rows
    h,w=rgb.shape[:2]; boxes=[]
    for xa,xb in yy_columns(rgb):
        panel=rgb[:,xa:xb]
        gray=panel.mean(2)
        background=np.percentile(gray,80,axis=1)
        contrast=background[:,None]-gray
        positive=contrast[contrast>3]
        threshold=max(18,float(np.percentile(positive,40))*.65) if positive.size else 18
        ink=contrast>threshold
        # Scrollbars can bridge many member rows. Remove persistent vertical
        # strokes only from the segmentation mask, never from the saved crop.
        live=np.zeros(ink.shape[1],int); longest=live.copy()
        for row in ink:
            live=np.where(row,live+1,0)
            longest=np.maximum(longest,live)
        ink[:,longest>max(20,h*.12)]=False
        edge=max(1,(xb-xa)//100)
        ink[:,:edge]=False; ink[:,-edge:]=False
        # Derive a recurring leading-icon lane. This excludes stray controls in
        # the left margin and trailing badges that can connect neighbouring rows.
        chroma=panel.max(2).astype(int)-panel.min(2)
        color_profile=(chroma>40).mean(0)
        lanes=[(a,b) for a,b in runs(color_profile>max(.10,color_profile.max()*.4)) if b-a>=2]
        if lanes:
            start,end=lanes[0]
            lane_end=min(panel.shape[1],start+max((end-start)*4,int(panel.shape[1]*.12)))
            occupancy=ink[:,start:lane_end].mean(1)
        else:
            occupancy=ink.mean(1)
        active=occupancy>max(.12 if lanes else .018,float(np.percentile(occupancy,20))*1.25)
        bands=runs(bridge(active,max(1,h//900)))
        bands=[(a,b) for a,b in bands if b-a>=max(3,h*.004)]
        if not bands:
            continue
        text_h=float(np.median([b-a for a,b in bands]))
        bands=runs(bridge(active,max(1,int(text_h*.20))))
        bands=[(a,b) for a,b in bands if b-a>=max(3,text_h*.45)]
        for i,(a,b) in enumerate(bands):
            top=max(0,a-int(text_h*.3)); bottom=min(h,b+int(text_h*.3))
            if i:
                top=max(top,(bands[i-1][1]+a)//2)
            if i+1<len(bands):
                bottom=min(bottom,(b+bands[i+1][0])//2)
            boxes.append((xa,top,xb,bottom))
    return boxes


def detect_regions(image,source):
    if image is None or not image.size:
        return []
    rgb=image[:,:,::-1]; h,w=image.shape[:2]
    boxes=game_boxes(rgb) if source=="游戏" else yy_boxes(rgb)
    result=[dict(box=tuple(map(int,box)),reliable=True,
                 method="局部卡片块分割" if source=="游戏" else "独立面板逐行分割")
            for box in sorted(boxes,key=lambda q:(q[1],q[0]))]
    result.append(dict(box=(0,0,w,h),reliable=False,method="原图兜底：核对切图遗漏"))
    return result




