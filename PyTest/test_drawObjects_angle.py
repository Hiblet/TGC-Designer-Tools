import cv2, numpy as np, math, pytest

class DummyPC:
    def tgcToCV2(self, x, z, image_scale):
        return int(z / image_scale), int(x / image_scale)

def draw_ellipse_like_objects(im, pc, x, z, sx, sz, ry, use_radians=False):
    image_scale = 1.0
    center_cv = pc.tgcToCV2(x, z, image_scale)
    center = (center_cv[1], center_cv[0])
    width  = sx / image_scale
    height = sz / image_scale
    if use_radians:
        rotation = - ry * math.pi / 180.0  # BUG: radians passed as degrees
    else:
        rotation = -float(ry)              # FIX: degrees (clockwise)
    cv2.ellipse(im, (center, (width, height), rotation), (255,255,255), thickness=-1, lineType=cv2.LINE_AA)

def measured_angle_deg(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
    return angle % 180.0

def ang_close(a, b, tol=5.0):
    # shortest distance on a circle of 180 degrees
    d = abs(((a - b + 90) % 180) - 90)
    return d <= tol

@pytest.mark.parametrize("ry", [15, 30, 60, 90, 120])
def test_objects_rotation_convention(ry):
    H, W = 500, 500
    pc = DummyPC()
    x, z = 250, 250
    sx, sz = 280, 80  # width >= height -> use the (90 + rotation) rule

    # Buggy draw (radians)
    bug = np.zeros((H, W, 3), np.uint8)
    draw_ellipse_like_objects(bug, pc, x, z, sx, sz, ry, use_radians=True)
    bug_angle = measured_angle_deg(bug)

    # Fixed draw (degrees)
    fix = np.zeros((H, W, 3), np.uint8)
    draw_ellipse_like_objects(fix, pc, x, z, sx, sz, ry, use_radians=False)
    fix_angle = measured_angle_deg(fix)

    rotation_cw_deg = -float(ry)
    if sx >= sz:
        expected = (90.0 + rotation_cw_deg) % 180.0   # == (90 - ry) % 180
    else:
        expected = (rotation_cw_deg) % 180.0

    assert ang_close(fix_angle, expected), f"fixed angle {fix_angle:.2f} not close to expected {expected:.2f} for ry={ry}"
    assert not ang_close(bug_angle, expected), f"bug angle {bug_angle:.2f} unexpectedly close to expected {expected:.2f} for ry={ry}"



def unwrap180(a):
    # map to a continuous line so we can compare differences
    return a  # good enough for small deltas in this test

def test_fixed_changes_with_ry_but_bug_does_not():
    pc = DummyPC()
    H, W = 400, 400
    x, z = 200, 200
    sx, sz = 280, 80

    def angle_for(ry, use_radians):
        im = np.zeros((H, W, 3), np.uint8)
        draw_ellipse_like_objects(im, pc, x, z, sx, sz, ry, use_radians=use_radians)
        return measured_angle_deg(im)

    a1_fix = unwrap180(angle_for(10, False))
    a2_fix = unwrap180(angle_for(40, False))
    a1_bug = unwrap180(angle_for(10, True))
    a2_bug = unwrap180(angle_for(40, True))

    # Fixed should change by ~30 deg (allow slack for sampling AA etc.)
    assert abs(((a2_fix - a1_fix + 90) % 180) - 90) >= 20.0

    # Bug should change very little (radians mistaken as degrees)
    assert abs(((a2_bug - a1_bug + 90) % 180) - 90) <= 5.0