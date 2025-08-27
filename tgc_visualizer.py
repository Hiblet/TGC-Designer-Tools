import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import overpy
import sys

from GeoPointCloud import GeoPointCloud
import tgc_definitions
import tgc_tools

def drawBrushesOnImage(brushes, color, im, pc, image_scale, fill=True):
    if not brushes:
        return

    for brush in brushes:
        # Guard required sub-objects
        pos = brush.get("position")
        scl = brush.get("scale")
        rot = brush.get("rotation")
        if not (pos and scl and rot):
            continue

        # Pull required scalars; skip if any missing
        x = pos.get("x"); z = pos.get("z")
        sx = scl.get("x"); sz = scl.get("z")
        ry = rot.get("y")
        if None in (x, z, sx, sz, ry):
            continue

        # Skip non-positive sizes (degenerate)
        if not (isinstance(sx, (int, float)) and isinstance(sz, (int, float)) and sx > 0 and sz > 0):
            continue

        # Convert to image coords
        center_cv = pc.tgcToCV2(x, z, image_scale)
        center = (center_cv[1], center_cv[0])  # (col,row) order for OpenCV

        width  = sx / image_scale
        height = sz / image_scale
        rotation = -ry  # OpenCV uses clockwise degrees

        thickness = -1 if fill else 4

        # Resolve brush type name safely
        try:
            brush_type_id = int(brush.get("type", -1))
        except (TypeError, ValueError):
            brush_type_id = -1
        brush_type_name = tgc_definitions.brushes.get(brush_type_id, "unknown")

        if 'square' in brush_type_name:
            # Squares seem to be larger than circles
            box_points = cv2.boxPoints((center, (2.0 * width, 2.0 * height), rotation))
            box_points = np.int32([box_points])  # fillPoly needs explicit int32
            if fill:
                cv2.fillPoly(im, box_points, color, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(im, box_points, True, color, thickness, lineType=cv2.LINE_AA)
        else:
            # Draw as ellipse for now; circles scale like radius
            bounding_box = (center, (1.414 * width, 1.414 * height), rotation)
            cv2.ellipse(im, bounding_box, color, thickness=thickness, lineType=cv2.LINE_AA)            

def drawSplinesOnImage(splines, color, im, pc, image_scale, course_version):
    if not splines:
        return

    dim2 = "z" if course_version == 25 else "y"

    for s in splines:
        wps = s.get("waypoints", [])
        if not wps:
            continue

        # Collect control points; skip malformed waypoints
        nds = []
        for wp in wps:
            p = wp.get("waypoint")
            if not p:
                continue
            x = p.get("x"); y2 = p.get(dim2)
            if x is None or y2 is None:
                continue
            nds.append(pc.tgcToCV2(x, y2, image_scale))

        # Need at least 2 points to draw a line; 3 for fill
        if len(nds) < 2:
            continue

        # Uses point coords not pixel indices → flip to (col,row)
        nds = np.array(nds)
        nds[:, [0, 1]] = nds[:, [1, 0]]
        nds = np.int32([nds])  # OpenCV wants int32

        # Thickness: default to image_scale, ensure >= 1
        try:
            thickness = int(s.get("width", image_scale))
        except (TypeError, ValueError):
            thickness = int(image_scale)
        if thickness < int(max(1, image_scale)):
            thickness = int(max(1, image_scale))

        is_filled = bool(s.get("isFilled")) or (s.get("state") == 3)
        if is_filled and len(nds[0]) >= 3:
            cv2.fillPoly(im, nds, color, lineType=cv2.LINE_AA)
        else:
            # isClosed can be a bool or inferred from state
            is_closed = s.get("isClosed")
            if is_closed is None:
                # Fall back: treat nonzero state as closed if that matched prior behavior
                is_closed = bool(s.get("state"))
            cv2.polylines(im, nds, is_closed, color, thickness, lineType=cv2.LINE_AA)

def drawObjectsOnImage(objects, color, im, pc, image_scale):
    if not objects:
        return

    for ob in objects:
        val = ob.get("Value", {})

        # Items
        for item in val.get("items", []):
            pos = item.get("position")
            scl = item.get("scale")
            rot = item.get("rotation", {})
            if not (pos and scl):
                continue

            x = pos.get("x"); z = pos.get("z")
            sx = scl.get("x"); sz = scl.get("z")
            if None in (x, z, sx, sz):
                continue

            center_cv = pc.tgcToCV2(x, z, image_scale)
            center = (center_cv[1], center_cv[0])  # point coords (col,row)

            try:
                width  = max(sx / image_scale, 8.0)
                height = max(sz / image_scale, 8.0)
            except Exception:
                continue

            ry = rot.get("y", 0)

            try:
                ry = float(ry)
            except (TypeError, ValueError):
                continue  # skip malformed rotation
            rotation = -ry  # degrees, clockwise for OpenCV

            bounding_box = (center, (width, height), rotation)
            cv2.ellipse(im, bounding_box, color, thickness=-1, lineType=cv2.LINE_AA)

        # Clusters
        for cluster in val.get("clusters", []):
            pos = cluster.get("position")
            rot = cluster.get("rotation", {})
            if not pos:
                continue

            x = pos.get("x"); z = pos.get("z")
            radius = cluster.get("radius")
            if None in (x, z, radius):
                continue

            center_cv = pc.tgcToCV2(x, z, image_scale)
            center = (center_cv[1], center_cv[0])

            try:
                width = height = radius / image_scale
            except Exception:
                continue

            ry = rot.get("y", 0)
            # NOTE: keeping existing behavior: rotation converted to radians
            rotation = - ry * math.pi / 180.0

            bounding_box = (center, (width, height), rotation)
            cv2.ellipse(im, bounding_box, color, thickness=-1, lineType=cv2.LINE_AA)

def drawHolesOnImage(holes, color, im, pc, image_scale, course_version):
    if course_version not in tgc_definitions.version_tags:
        return
    
    tee_tag = tgc_definitions.version_tags[course_version]['tees']

    if not holes:
        return

    for h in holes:
        # Waypoints (need at least 2 to draw lines)
        wps = h.get("waypoints", [])
        waypoints = []
        for wp in wps:
            x = wp.get("x"); z = wp.get("z")
            if x is None or z is None:
                continue
            waypoints.append(pc.tgcToCV2(x, z, image_scale))

        if len(waypoints) < 2:
            continue  # nothing to draw for this hole

        # Tees (2K25 tees may be wrapped in {"position": {...}})
        tees = []
        for t in h.get(tee_tag, []):
            tp = t.get("position", t) if course_version == 25 else t
            x = tp.get("x"); z = tp.get("z")
            if x is None or z is None:
                continue
            tees.append(pc.tgcToCV2(x, z, image_scale))

        # Flip to (col,row) for OpenCV
        waypoints = np.array(waypoints, dtype=np.int32)
        waypoints[:, [0, 1]] = waypoints[:, [1, 0]]
        if tees:
            tees = np.array(tees, dtype=np.int32)
            tees[:, [0, 1]] = tees[:, [1, 0]]

        # Draw a line between each waypoint
        thickness = 5
        for i in range(len(waypoints) - 1):
            first_point  = tuple(waypoints[i])
            second_point = tuple(waypoints[i + 1])
            cv2.line(im, first_point, second_point, color, thickness=thickness, lineType=cv2.LINE_AA)

        # Draw a line from each tee to the second waypoint (index 1 exists because len>=2)
        first_waypoint = tuple(waypoints[1])
        for tee in tees:
            cv2.line(im, tuple(tee), first_waypoint, color, thickness=thickness, lineType=cv2.LINE_AA)

def drawCourseAsImage(course_json, course_version):
    im = np.zeros((2000, 2000, 3), np.float32)  # Courses are 2000m x 2000m
    image_scale = 1.0  # Draw one pixel per meter
    pc = GeoPointCloud()
    pc.width = 2000.0
    pc.height = 2000.0

    if course_version not in tgc_definitions.version_tags:
        return

    hole_tag   = tgc_definitions.version_tags[course_version]['holes']
    crowd_tag  = tgc_definitions.version_tags[course_version]['crowd']
    spline_tag = tgc_definitions.version_tags[course_version]['splines']
    surface_tag= tgc_definitions.version_tags[course_version]['surfaces']
    oob_tag    = tgc_definitions.version_tags[course_version]['oob']
    obj_tag    = tgc_definitions.version_tags[course_version]['objects']

    # Pick layer container (reads only, so .get is fine)
    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})

    # Draw terrain first
    drawBrushesOnImage(layer_json.get("terrainHeight", []), (0.35, 0.2, 0.0), im, pc, image_scale)
    drawBrushesOnImage(layer_json.get("height", []),        (0.5,  0.2755, 0.106), im, pc, image_scale)

    # Surfaces (2K25 may wrap as {"brushes":[...]})
    uls = layer_json.get(surface_tag, [])
    if course_version == 25 and isinstance(uls, dict):
        uls = uls.get("brushes", [])

    # Splines may live at root or layer
    ss = (course_json.get(spline_tag, []) or layer_json.get(spline_tag, []))

    # Real water (2K25 may wrap)
    water = layer_json.get("water", [])
    if course_version == 25 and isinstance(water, dict):
        water = water.get("brushes", [])
    water_color = (0.1, 0.2, 0.5)
    drawBrushesOnImage(water, water_color, im, pc, image_scale)

    
    # Helper selectors with .get(...) to avoid KeyErrors
    def spl(n):  # splines by surface id
        return [s for s in ss if isinstance(s, dict) and s.get("surface") == n]

    def br(n):   # brushes by surfaceCategory
        return [b for b in uls if isinstance(b, dict) and b.get("surfaceCategory") == n]

    # Mulch/Water Visualization Surface #2 (low priority)
    surface2_color = (0.1, 0.2, 0.25)
    drawSplinesOnImage(spl(8), surface2_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(8),  surface2_color, im, pc, image_scale)

    # Heavy rough
    heavy_rough_color = (0, 0.3, 0.1)
    drawSplinesOnImage(spl(4), heavy_rough_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(4),  heavy_rough_color, im, pc, image_scale)

    # Rough
    rough_color = (0.1, 0.35, 0.15)
    drawSplinesOnImage(spl(3), rough_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(3),  rough_color, im, pc, image_scale)

    # Fairways
    fairway_color = (0, 0.75, 0.2)
    drawSplinesOnImage(spl(2), fairway_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(2),  fairway_color, im, pc, image_scale)

    # Greens
    green_color = (0, 1.0, 0.2)
    drawSplinesOnImage(spl(1), green_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(1),  green_color, im, pc, image_scale)

    # Bunkers
    bunker_color = (0.85, 0.85, 0.7)
    drawSplinesOnImage(spl(0), bunker_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(0),  bunker_color, im, pc, image_scale)

    # Surface #1 - Gravel?
    surface1_color = (0.7, 0.7, 0.7)
    drawSplinesOnImage(spl(7), surface1_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(7),  surface1_color, im, pc, image_scale)

    # Surface #3 - Cart Path
    cart_path_color = (0.3, 0.3, 0.3)
    drawSplinesOnImage(spl(10), cart_path_color, im, pc, image_scale, course_version)
    drawBrushesOnImage(br(10),  cart_path_color, im, pc, image_scale)

    # Out of bounds (white) — 2K25 may wrap
    out_of_bounds_color = (1.0, 1.0, 1.0)
    oob_json = layer_json.get(oob_tag, [])
    if course_version == 25 and isinstance(oob_json, dict):
        oob_json = oob_json.get("brushes", [])
    drawBrushesOnImage(oob_json, out_of_bounds_color, im, pc, image_scale, fill=False)

    # Crowds (pink) — 2K25 may wrap
    crowd_color = (1.0, 0.4, 0.75)
    crowd_json = layer_json.get(crowd_tag, [])
    if course_version == 25 and isinstance(crowd_json, dict):
        crowd_json = crowd_json.get("brushes", [])
    drawBrushesOnImage(crowd_json, crowd_color, im, pc, image_scale, fill=False)

    # Objects — may live at root or layer
    object_color = (0.95, 0.9, 0.2)
    objs = (course_json.get(obj_tag, []) or layer_json.get(obj_tag, []))
    drawObjectsOnImage(objs, object_color, im, pc, image_scale)

    # Holes (root)
    hole_color = (0.9, 0.3, 0.2)
    drawHolesOnImage(course_json.get(hole_tag, []), hole_color, im, pc, image_scale, course_version)

    return im

if __name__ == "__main__":
    print("main")

    if len(sys.argv) < 2:
        print("Usage: python program.py COURSE_DIRECTORY")
        sys.exit(0)
    else:
        lidar_dir_path = sys.argv[1]

    print("Loading course file")
    course_json = tgc_tools.get_course_json(lidar_dir_path)

    im = drawCourseAsImage(course_json)

    fig = plt.figure()

    plt.imshow(im, origin='lower')

    plt.show()
