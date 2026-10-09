"""Find a YY nickname view inside an already materialized complete member row."""
import numpy as np
from attendance_tool_v7_1.core.adaptive_regions import runs,bridge


def yy_name_view(crop):
    h,w=crop.shape[:2]
    rgb=crop[:,:,::-1].astype(float)
    chroma=rgb.max(2)-rgb.min(2)
    colored=chroma>40
    profile=colored.mean(0)
    lanes=[(a,b) for a,b in runs(bridge(profile>.20,max(1,h//20)))
           if b-a>=max(3,h*.20) and a<w*.6]
    if not lanes:
        return None
    start,end=lanes[0]
    # Start after the recurring icon. No fixed x-coordinate or nickname length.
    left=int(end+max(1,h*.05))
    if w-left<h:
        return None
    sample=rgb[:,left:min(w,left+max(3,int(h*.85)))]
    sample_chroma=sample.max(2)-sample.min(2)
    dark=(sample.mean(2)<150)&(sample_chroma<40)
    if dark.sum()>max(3,h*.12):
        tail=colored
    else:
        strong=sample_chroma>40
        if not strong.any():
            return None
        normalized=sample/np.maximum(sample.sum(2,keepdims=True),1)
        name_color=np.median(normalized[strong],axis=0)
        colors=rgb/np.maximum(rgb.sum(2,keepdims=True),1)
        tail=colored&(np.abs(colors-name_color).sum(2)>.40)
    tail_profile=tail.mean(0)
    right=w
    for a,b in runs(bridge(tail_profile>.15,max(1,h//25))):
        if a-left>=h*.80 and b-a>=max(2,h*.10):
            right=int(a-max(1,h*.04))
            break
    if right-left<h*.8:
        return None
    return crop[:,left:right].copy(),(left,0,right,h)
