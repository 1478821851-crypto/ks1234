"""Find member rows from recurring leading role icons and status/avatar anchors.

Blank space inside a row is never a panel boundary. Measurements are relative
to the pane and observed icons, so pasted sections can change scale/indentation.
"""
import cv2
import numpy as np


def member_rows(rgb):
    h, w = rgb.shape[:2]
    # Wide collages use the existing independent-panel detector.
    if h < w * 2 or w < 40:
        return None
    pixels = rgb.astype(float)
    gray = pixels.mean(2)
    chroma = pixels.max(2) - pixels.min(2)
    color = ((chroma > 40) & (pixels.max(2) > 85)).astype(np.uint8)
    color[:, int(w * .32):] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(color, 8)
    candidates = []
    for x, y, bw, bh, area in stats[1:]:
        if (w * .028 <= bw <= w * .095 and w * .032 <= bh <= w * .10
                and area >= w * w * .00045 and x > w * .12):
            candidates.append(tuple(map(int, (x, y, bw, bh))))
    clusters = []
    for item in sorted(candidates):
        group = next((g for g in clusters if abs(item[0] - np.median([r[0] for r in g])) <= w * .012
                      and .72 <= item[2]/np.median([r[2] for r in g]) <= 1.38
                      and .72 <= item[3]/np.median([r[3] for r in g]) <= 1.38), None)
        if group is None:
            clusters.append([item])
        else:
            group.append(item)
    clusters = [g for g in clusters if len(g) >= 3]
    roles = []
    for group in clusters:
        width, height = np.median([r[2:] for r in group], axis=0)
        for item in group:
            x, y, bw, bh = item
            if not (.65 * width <= bw <= 1.4 * width and .65 * height <= bh <= 1.4 * height):
                continue
            scores = [sum(abs(r[1] - y) < height * 8 for r in g) * np.median([r[3] for r in g]) for g in clusters]
            if scores[clusters.index(group)] >= max(scores) * .8:
                roles.append(item)
    if len(roles) < 5:
        return None
    # Each role is one row; nearby components (e.g. phone flags) are evidence,
    # never another crop. Keep the largest role-shaped component per row.
    roles.sort(key=lambda r: r[1])
    merged = []
    for role in roles:
        if merged and abs((role[1] + role[3]/2) - (merged[-1][1] + merged[-1][3]/2)) < min(role[3], merged[-1][3]) * .65:
            if role[2] * role[3] > merged[-1][2] * merged[-1][3]:
                merged[-1] = role
        else:
            merged.append(role)
    roles = merged
    bg = np.percentile(gray, 85, axis=1)
    ink = ((bg[:, None] - gray > 10) | ((chroma > 30) & (pixels.max(2) > 70))).astype(np.uint8)
    ink[:, int(w * .24):] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    anchors = [tuple(map(int, r[:4])) for r in stats[1:] if
               max(3, w*.012) <= r[2] <= w*.12 and max(3, w*.012) <= r[3] <= w*.14
               and .45 <= r[2]/r[3] <= 2 and r[4] >= w*w*.00015 and r[0] > w*.025]
    pairs = []
    for role in roles:
        x,y,bw,bh = role
        nearby = [a for a in anchors if a[0]+a[2] <= x+1 and x-a[0] < w*.14
                  and abs(a[1]+a[3]/2-y-bh/2) < bh*.45]
        for anchor in nearby:
            pairs.append((role, anchor))
    # Recover white/low-saturation role icons using the same local anchor lane.
    for a in anchors:
        cy = a[1]+a[3]/2
        if cy < roles[0][1]:
            continue
        if any(abs(cy-r[1]-r[3]/2) < r[3]*.65 for r in roles):
            continue
        local = sorted(pairs, key=lambda p: abs(p[0][1]+p[0][3]/2-cy))[:8]
        matches = [(r,b) for r,b in local if abs(a[0]-b[0]) <= w*.015]
        if not matches:
            continue
        anchor_height = float(np.median([b[3] for _,b in matches]))
        if not .70*anchor_height <= a[3] <= 1.4*anchor_height:
            continue
        r,b = min(matches, key=lambda p: abs(p[0][1]-cy))
        if abs(r[1]+r[3]/2-cy) > r[3]*20:
            continue
        x = r[0]+a[0]-b[0]
        top, bottom = max(0,int(cy-r[3]*.65)), min(h,int(cy+r[3]*.65)+1)
        # Above mask stops at the leading area, but faint role icons may be
        # farther right after indentation. Inspect the original contrast there.
        lane = (bg[top:bottom,None]-gray[top:bottom,x:x+r[2]] > 5)
        occupied = np.flatnonzero(lane.sum(1) >= max(2,r[2]*.18))
        if len(occupied) >= r[3]*.45:
            roles.append((x,top+int(occupied[0]),r[2],int(occupied[-1]-occupied[0]+1)))
        else:
            text = bg[top:bottom,None]-gray[top:bottom,x+r[2]:int(w*.70)] > 35
            text_rows = np.flatnonzero(text.sum(1) >= max(3,r[2]*.25))
            if len(text_rows) >= r[3]*.45:
                roles.append((x,int(round(cy-r[3]/2)),r[2],r[3]))
    roles.sort(key=lambda r:r[1])
    median_height = float(np.median([r[3] for r in roles]))
    roles = [r for r in roles if not (r[1] <= 1 and r[3] < median_height*.9)]
    median_width = float(np.median([r[2] for r in roles]))
    leading = min((a[0] for _,a in pairs),default=0)
    crop_left = int(leading-median_width) if leading > median_width*2 else 0
    boxes = []
    for i,(x,y,bw,bh) in enumerate(roles):
        pad = max(2,int(round(bh*.5)))
        top, bottom = max(0,y-pad), min(h,y+bh+pad)
        if i == 0:
            top = max(top,y-max(2,int(bh*.25)))
        local_height = float(np.median([r[3] for r in roles[max(0,i-3):i+4]]))
        if bh < local_height*.85:
            # A floating channel toolbar can cover the top of a pasted row.
            # Preserve its visible name without adding the toolbar to the crop.
            top = max(top,y-2)
        if i:
            top=max(top,(roles[i-1][1]+roles[i-1][3]+y)//2)
        if i+1<len(roles):
            bottom=min(bottom,(y+bh+roles[i+1][1])//2)
        boxes.append((crop_left,top,w,bottom))
    return boxes


def member_panels(rgb, columns):
    """Recognize complete member panes; ignore narrow adjacent toolbars."""
    # A long vertical edge is a real pane divider. A blank gap between the
    # nickname and its badges is not. Resolve genuine dividers before measuring
    # icon sizes against the pane width; adjacent chat must not change the scale.
    h,w=rgb.shape[:2]
    difference=np.max(np.abs(np.diff(rgb.astype(np.int16),axis=1)),2)
    dividers=[int(right) for _,right in columns if right<w
              and float((difference[:,int(right)-1]>5).mean())>.35]
    if dividers:
        edges=[0]+dividers+[w]
        result=[]
        for left,right in zip(edges,edges[1:]):
            rows=member_rows(rgb[:,left:right])
            if rows is not None:
                result.extend(tuple(map(int,(a+left,b,c+left,d))) for a,b,c,d in rows)
        if result:
            return result
    rows = member_rows(rgb)
    if rows is not None:
        return rows
    result = []
    for left,right in columns:
        rows = member_rows(rgb[:,left:right])
        if rows is not None:
            result.extend(tuple(map(int,(a+left,b,c+left,d))) for a,b,c,d in rows)
    return result or None


