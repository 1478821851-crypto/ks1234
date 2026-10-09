import streamlit as st
import cv2
import json
from paddleocr import PaddleOCR, TextRecognition

@st.cache_resource
def load_ocr():
    # 云端轻量化：截图不需要文档方向分类、页面矫正、文本行方向分类。
    # 关闭这 3 个辅助模型，只保留真正需要的文字检测 + 中文识别模型，
    # 可显著降低 Streamlit Cloud 内存占用；姓名匹配与考勤规则完全不变。
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        cpu_threads=4,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


@st.cache_resource
def load_name_recognizer():
    return TextRecognition(model_name="PP-OCRv6_medium_rec",cpu_threads=4)


def name_evidence(result):
    records=[]
    for item in result:
        data=item.json
        if callable(data):
            data=data()
        if isinstance(data,str):
            data=json.loads(data)
        res=data.get('res',data)
        text=str(res.get('rec_text','')).strip()
        if text:
            records.append(dict(text=text,score=float(res.get('rec_score',0)),view='姓名视图'))
    return records


class RegionReader:
    """Recognize individual row-name views; retry unmatched full original crops."""
    def __init__(self,source,names,status=None):
        self.source=source;self.names=names;self.status=status
        self.cache={};self.name_boxes={}
        self.ocr=None;self.ocr_error=None

    def prepare(self,crops):
        if self.source!='YY':
            return
        from attendance_tool_v7_1.core.name_view import yy_name_view
        tasks=[]
        for crop in crops:
            view=yy_name_view(crop)
            if view:
                tasks.append((crop,view[0]))
                self.name_boxes[id(crop)]=view[1]
        if not tasks:
            return
        recognizer=load_name_recognizer()
        # Each input remains one distinct name view; no image mosaics or merged
        # OCR text. Batch outputs preserve one-to-one crop provenance.
        for base in range(0,len(tasks),8):
            group=tasks[base:base+8]
            if self.status:
                self.status.write(f"YY：逐行识别姓名 {base+1}–{min(base+8,len(tasks))}/{len(tasks)}")
            results=recognizer.predict([view for _,view in group],batch_size=8)
            if len(results)!=len(group):
                raise RuntimeError('姓名识别返回数与裁图数不一致，全部保留人工核对。')
            for (crop,_),result in zip(group,results):
                self.cache[id(crop)]=name_evidence([result])

    def __call__(self,crop):
        from attendance_tool_v7_1.core.pipeline import match_region
        direct=self.cache.get(id(crop),[])
        confirmed,_=match_region([r['text'] for r in direct],self.names)
        if confirmed and any(r['score']>=.90 for r in direct):
            return direct
        if self.ocr_error:
            raise RuntimeError(self.ocr_error)
        if self.ocr is None:
            try:
                self.ocr=load_ocr()
            except Exception as exc:
                self.ocr_error=f'OCR初始化失败：{exc}'
                raise RuntimeError(self.ocr_error) from exc
        full=ocr_small_image(self.ocr,crop,evidence=True)
        return direct+full


def extract_texts(result, evidence=False):

    texts = []

    if result is None:
        return texts

    for item in result:

        try:

            data = item.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                data = json.loads(data)

            if isinstance(data, dict):

                res = data.get("res", data)

                rec_texts = res.get(
                    "rec_texts",
                    []
                )

                scores = res.get("rec_scores", [])
                for index, text in enumerate(rec_texts):

                    text = str(text).strip()

                    if text:
                        score = float(scores[index]) if index < len(scores) else 0.0
                        texts.append({"text":text,"score":score} if evidence else text)

        except Exception:
            pass

    return texts


def prepare_ocr_image(image):
    """Exactly the V6.6.1 resize and border; preserve BGR uint8 pixels."""
    h, w = image.shape[:2]
    scale = min(4.0, max(1.0, 128 / h), 2200 / w)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)


def ocr_small_image(ocr, image, evidence=False):
    if image is None or image.size == 0:
        return []
    try:
        # Paddle accepts BGR ndarray. Lossless PNG decoding in V6.6.1 produces
        # the same pixels. Keep prediction order and batch size unchanged.
        result = ocr.predict(prepare_ocr_image(image))
        return extract_texts(result, evidence=evidence)
    except Exception as exc:
        raise RuntimeError(f"裁图 OCR 失败：{exc}") from exc
