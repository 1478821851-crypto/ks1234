"""V6.6: freeze crops first, then OCR and conservatively match each crop."""
import re
import unicodedata
from difflib import SequenceMatcher
from attendance_tool_v7_1.core.adaptive_regions import detect_regions, grid_boxes
try:
    from opencc import OpenCC
    _converter = OpenCC('t2s')
except ImportError:
    _converter = None


def identity(text):
    text = unicodedata.normalize("NFKC", str(text))
    if _converter:
        text = _converter.convert(text)
    # Decorative glyphs may be omitted by OCR. Every canonical identity is
    # checked against the entire roster, so collisions remain manual candidates.
    text = text.replace('ノ', '丿')
    text = re.sub(r"[\s♡♥❤★☆♛♕♚♔✓✔☑♪♫♬•·●○◎◇◆♢♦♧♣丶丷丿ˇ^`'\"\\|/_-]", "", text)
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith(('P','S')))
    return text.strip()


def text_variants(text):
    raw=str(text).strip()
    base=[raw,re.sub(r"^(?:【[^】]{1,20}】|\[[^\]]{1,20}\])\s*",'',raw)]
    result=[]
    for item in base:
        result += [item,re.split(r"https?://|www\.|[（(【\[]",item,maxsplit=1)[0].strip(),
                   re.split(r"[：:]",item,maxsplit=1)[0].strip()]
    return list(dict.fromkeys(v for v in result if v))


def match_region(texts, names):
    exact = set()
    for text in texts:
        for v in text_variants(text):
            matches = [n for n in names if identity(n) == identity(v) and identity(n)]
            exact.update(matches)
    # One physical card/row must never sign in multiple different people.
    confirmed = next(iter(exact)) if len(exact) == 1 else None
    def suggestion_score(name):
        target=identity(name)
        variants=[identity(v) for text in texts for v in text_variants(text)]
        scores=[SequenceMatcher(None,target,v).ratio() for v in variants]
        # Substrings influence suggestions only, never automatic attendance.
        scores += [.95 for v in variants if len(target)>=2 and target in v]
        return max(scores or [0])
    scored = sorted(((suggestion_score(n),n) for n in names), reverse=True)
    suggestions = [{"标准姓名": n, "最高相似度": round(score*100, 1),
                    "OCR最接近": " / ".join(texts), "说明": "仅供人工确认"}
                   for score, n in scored[:3] if score >= .45]
    return confirmed, suggestions


def recognize_regions(image, source, names, status=None, manual=None, reader=None):
    regions = grid_boxes(image, *manual) if manual else detect_regions(image, source)
    if manual:
        h, w = image.shape[:2]
        regions.append(dict(box=(0,0,w,h), reliable=False, method="手动切图原图兜底"))
    # Materialize original crops before any OCR/model initialization.
    for i, region in enumerate(regions):
        x1,y1,x2,y2 = region["box"]
        region.update(id=i+1, crop=image[y1:y2,x1:x2].copy(), texts=[],
                      matched=None, suggestions=[], error=None, evidence=[])
    if reader is None:
        try:
            from attendance_tool_v7_1.core.ocr_engine import RegionReader
            reader = RegionReader(source,names,status)
        except Exception as exc:
            for region in regions:
                region["error"] = f"OCR 初始化失败：{exc}"
            return set(), [], [], regions
    if hasattr(reader,'prepare'):
        try:
            reader.prepare([r['crop'] for r in regions if r['reliable']])
            for r in regions:
                r['name_box'] = getattr(reader,'name_boxes',{}).get(id(r['crop']))
        except Exception as exc:
            for r in regions:
                r['error'] = f'姓名视图识别失败，将尝试完整裁图：{exc}'
    found, uncertain, texts = set(), [], []
    tasks=[r for r in regions if r['reliable']]
    for i, region in enumerate(tasks):
        if status:
            status.write(f"{source}：逐张识别 {i+1}/{len(tasks)}")
        try:
            raw = list(reader(region["crop"]))
            region["evidence"] = [entry if isinstance(entry,dict) else {"text":str(entry),"score":None} for entry in raw]
            region["texts"] = [entry["text"] for entry in region["evidence"]]
            region["matched"], region["suggestions"] = match_region(region["texts"], names)
            if region["matched"]:
                # A correct-looking but low-confidence OCR string remains manual.
                verified = any((entry.get("score") is None or entry["score"] >= .90)
                    and match_region([entry["text"]],names)[0] == region["matched"]
                    for entry in region["evidence"])
                if not verified:
                    region["matched"] = None
            region['error'] = None
        except Exception as exc:
            region["error"] = str(exc)
        texts.extend(region["texts"])
        if region["matched"]:
            found.add(region["matched"])
        else:
            uncertain.extend(dict(row, 区域=region["id"]) for row in region["suggestions"])
    return found, uncertain, list(dict.fromkeys(texts)), regions
