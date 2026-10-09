import numpy as np
import cv2

def read_image(uploaded_file):

    if uploaded_file is None:
        return None

    data = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )
