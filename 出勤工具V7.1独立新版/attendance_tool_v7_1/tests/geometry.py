import numpy as np

def check_rows(boxes, case, width, height):
    assert len(boxes) == case['count']
    assert len({tuple(b) for b in boxes}) == len(boxes)
    if 'name_boxes' in case:
        remaining=list(boxes)
        for rect in case['name_boxes']:
            rect=[round(v*(width/case['width'] if i%2==0 else height/case['height'])) for i,v in enumerate(rect)]
            matching=[b for b in remaining if b[0]<=rect[0] and b[1]<=rect[1] and b[2]>=rect[2] and b[3]>=rect[3]]
            assert len(matching)==1,(rect,matching)
            remaining.remove(matching[0])
        for i,b in enumerate(boxes):
            assert 0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height
            for other in boxes[i+1:]:
                assert min(b[2],other[2])<=max(b[0],other[0]) or min(b[3],other[3])<=max(b[1],other[1])
        if case['key'] in ('G','I'):
            assert all(b[2]<=round(310*width/case['width']) for b in boxes)
        elif case['key']=='H':
            assert sum(b[0]<width*.4 for b in boxes)==37
            assert sum(b[0]>width*.45 for b in boxes)==29
        return
    for i, (box, band) in enumerate(zip(boxes, case['name_bands'])):
        top, bottom = [round(y * height / case['height']) for y in band]
        assert box[0] == 0 and width*.75 <= box[2] <= width, (i, box)
        # Independent name-band annotations check coverage, not just counts.
        assert 0 <= box[1] <= top < bottom <= box[3] <= height, (i, box, band)
        assert box[3]-box[1] <= (bottom-top)*3.8 + 3
        if i:
            assert boxes[i-1][3] <= box[1]



def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def assert_card_boundaries(actual, expected, width, height, min_iou=.94):
    """One-to-one location/size checks catch misses hidden by equal counts."""
    assert len(actual) == len(expected)
    assert len({tuple(box) for box in actual}) == len(actual)
    remaining = list(actual)
    for card in expected:
        scores = [intersection(box, card) /
                  (area(box) + area(card) - intersection(box, card))
                  for box in remaining]
        best = int(np.argmax(scores))
        assert scores[best] >= min_iou, (card, remaining[best], scores[best])
        matched = remaining[best]
        for axis in range(4):
            extent = card[2] - card[0] if axis % 2 == 0 else card[3] - card[1]
            assert abs(matched[axis] - card[axis]) <= max(3, extent * .04)
        remaining.pop(best)
    for i, box in enumerate(actual):
        assert 0 <= box[0] < box[2] <= width
        assert 0 <= box[1] < box[3] <= height
        for other in actual[i + 1:]:
            assert intersection(box, other) < .05 * min(area(box), area(other))


