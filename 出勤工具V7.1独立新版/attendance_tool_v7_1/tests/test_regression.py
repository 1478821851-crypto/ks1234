import hashlib,json,unittest
from pathlib import Path
import cv2,numpy as np
from attendance_tool_v7_1.core.adaptive_regions import detect_regions,grid_boxes
from attendance_tool_v7_1.core.pipeline import match_region,recognize_regions
from attendance_tool_v7_1.tests.geometry import check_rows,assert_card_boundaries
FIXTURES=Path(__file__).parent/'fixtures'
class SegmentationTests(unittest.TestCase):
    pass

def make_case(case,source,scale):
    def test(self):
        folder=FIXTURES/('yy_segmentation' if source=='YY' else 'game_segmentation')
        data=(folder/case['file']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),case['sha256'])
        image=cv2.imdecode(np.frombuffer(data,np.uint8),1);oh,ow=image.shape[:2]
        if scale!=1:image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        h,w=image.shape[:2];regions=detect_regions(image,source);boxes=[r['box'] for r in regions if r['reliable']]
        json.dumps(regions)
        if source=='YY':check_rows(boxes,case,w,h)
        else:
            expected=[[round(v*(w/ow if i%2==0 else h/oh)) for i,v in enumerate(b)] for b in case['boxes']]
            assert_card_boundaries(boxes,expected,w,h,min_iou=.92 if scale<1 else .94)
        self.assertFalse(regions[-1]['reliable'])
        self.assertEqual(regions[-1]['box'],(0,0,w,h))
    return test
for source,folder in [('YY','yy_segmentation'),('游戏','game_segmentation')]:
    cases=json.loads((FIXTURES/folder/'annotations.json').read_text(encoding='utf-8'))
    for case in cases:
        for scale in (1.,.7,1.3):
            setattr(SegmentationTests,f"test_{folder}_{case['key']}_{str(scale).replace('.','_')}",make_case(case,source,scale))

class PipelineTests(unittest.TestCase):
    def test_exact_and_fuzzy_matching(self):
        self.assertEqual(match_region(['鸑鷟'],['鸑鷟','其他'])[0],'鸑鷟')
        self.assertIsNone(match_region(['神酱大王'],['神酱王'])[0])
    def test_ambiguous_decoration_never_auto_confirms(self):
        self.assertIsNone(match_region(['阿·明'],['阿明','阿·明'])[0])
    def test_low_confidence_and_original_crop(self):
        image=np.full((60,180,3),180,np.uint8)
        seen=[]
        def reader(crop):
            seen.append(crop.copy());return [{'text':'成员甲','score':.50}]
        found,_,_,regions=recognize_regions(image,'YY',['成员甲'],manual=((0,0,100,100),1,1),reader=reader)
        self.assertEqual(found,set());self.assertEqual(len(seen),1)
        np.testing.assert_array_equal(seen[0],image)
        self.assertIsNone(regions[0]['matched']);self.assertFalse(regions[-1]['reliable'])
    def test_confident_exact_and_failed_reader(self):
        image=np.full((60,180,3),180,np.uint8)
        found,_,_,regions=recognize_regions(image,'YY',['成员甲'],manual=((0,0,100,100),1,1),reader=lambda crop:[{'text':'成员甲','score':.99}])
        self.assertEqual(found,{'成员甲'})
        def failed(crop):raise RuntimeError('test error')
        found,_,_,regions=recognize_regions(image,'YY',['成员甲'],manual=((0,0,100,100),1,1),reader=failed)
        self.assertFalse(found);self.assertEqual(regions[0]['error'],'test error')
if __name__=='__main__':unittest.main()
