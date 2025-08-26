import base64
import cv2
import gzip
import itertools
import json
import math
import numpy as np
import os
from pathlib import Path
import tgc_definitions

def base64GZDecode(data):
    gz_data = base64.b64decode(data)
    return gzip.decompress(gz_data)

def base64GZEncode(data, level):
    gz_data = gzip.compress(data, compresslevel=level)
    return base64.b64encode(gz_data)

def create_directory(course_directory):
    course_dir = Path(course_directory)
    try:
        Path.mkdir(course_dir, mode=0o777)
    except FileExistsError as err:
        pass
    return course_dir

def get_course_file(course_directory):
    # Find the course file
    course_list = list(Path(course_directory).glob('*.course'))
    if not course_list:
        print("No courses found in: " + course_directory)
        return None
    elif len(course_list) > 1:
        print("More than one course found, using " + str(course_list[0]))
    return course_list[0]

def get_course_name(course_directory):
    course_file = get_course_file(course_directory)
    if course_file is None:
        return None
    return course_file.stem

def unpack_course_file(course_directory, course_file=None):
    # Make directory structure
    course_dir = Path(course_directory)
    output_dir = course_dir / 'unpacked'
    description_dir = output_dir / 'course_description'
    metadata_dir = output_dir / 'metadata'
    thumbnail_dir = output_dir / 'thumbnail'

    try:
        Path.mkdir(output_dir, mode=0o777)
        Path.mkdir(description_dir, mode=0o777)
        Path.mkdir(metadata_dir, mode=0o777)
        Path.mkdir(thumbnail_dir, mode=0o777)
    except FileExistsError as err:
        pass

    course_name = ""
    if course_file is None:
        course_file = get_course_file(course_directory)
        course_name = get_course_name(course_directory)

    with gzip.open(str(course_file), 'r') as f:
        file_content = f.read()
        course_json = json.loads(file_content.decode('utf-16'))

        with (output_dir / 'full.json').open('w') as f:
            f.write(file_content.decode('utf-16'))

        course_description64 = course_json["binaryData"]["CourseDescription"]
        thumbnail64 = course_json["binaryData"]["Thumbnail"]
        course_metadata64 = course_json["binaryData"]["CourseMetadata"]

        course_description_json = base64GZDecode(course_description64).decode('utf-16')
        # Remove potential strange unicode characters like u200b
        course_description_json = (course_description_json.encode('ascii', 'ignore')).decode("utf-8")
        with (description_dir / 'course_description.json').open('w') as f:
            f.write(course_description_json)

        thumbnail_json = base64GZDecode(thumbnail64).decode('utf-16')
        # Remove potential strange unicode characters like u200b
        thumbnail_json = (thumbnail_json.encode('ascii', 'ignore')).decode("utf-8")
        t_json = json.loads(thumbnail_json)
        with (thumbnail_dir / 'thumbnail.json').open('w') as f:
            f.write(thumbnail_json)
        thumbnail_jpg = base64.b64decode(t_json["image"])
        with (thumbnail_dir / 'thumbnail.jpg').open('wb') as f:
            f.write(thumbnail_jpg)

        course_metadata_json = base64GZDecode(course_metadata64).decode('utf-16')
        # Remove potential strange unicode characters like u200b
        course_metadata_json = (course_metadata_json.encode('ascii', 'ignore')).decode("utf-8")
        with (metadata_dir / 'course_metadata.json').open('w') as f:
            f.write(course_metadata_json)        

    return course_name

def pack_course_file(course_directory, course_name=None, output_file=None, course_json=None, course_version=-1):
    if course_version not in tgc_definitions.version_tags:
        return

    file_encoding = tgc_definitions.version_tags[course_version]['file_encoding']    
    course_dir = Path(course_directory)

    output_path = None
    if output_file is not None:
        output_path = Path(output_file)
    else:
        if course_name is None:
            course_name = get_course_name(course_directory)
            if course_name is None:
                # Nothing found, just use 'output.course'
                course_name = 'output'
        output_path = course_dir / (course_name + '.course')

    print("Saving course as: " + str(output_path))

    # Write out new course description before packing into course
    if course_json is not None:
        write_course_json(course_directory, course_json)

    with (course_dir / 'unpacked/course_description/course_description.json').open('r') as desc:
        with (course_dir / 'unpacked/metadata/course_metadata.json').open('r') as meta:
            with (course_dir / 'unpacked/thumbnail/thumbnail.json').open('r') as thumb:
                desc_read = desc.read()
                meta_read = meta.read()
                thumb_read = thumb.read()
                desc_encoded = base64GZEncode(desc_read.encode(file_encoding), 1).decode('utf-8')
                meta_encoded = base64GZEncode(meta_read.encode(file_encoding), 1).decode('utf-8')
                thumb_encoded = base64GZEncode(thumb_read.encode(file_encoding), 1).decode('utf-8')

                output_json = json.loads('{"data":{},"binaryData":{}}')
                output_json["binaryData"]["CourseDescription"] = desc_encoded
                output_json["binaryData"]["Thumbnail"] = thumb_encoded
                output_json["binaryData"]["CourseMetadata"] = meta_encoded

                # Special dense encoding used for course files
                output_string = json.dumps(output_json, separators=(',', ':'))
 
                # Write to final gz format
                output_gz = gzip.compress(output_string.encode(file_encoding), 1)

                with (output_path).open('wb') as f:
                    f.write(output_gz)
                
                return str(output_path)

def get_course_json(course_directory):
    course_dir = Path(course_directory)
    course_json = ""
    with (course_dir / 'unpacked/course_description/course_description.json').open('r') as f:
        course_json = json.loads(f.read())  
        
    return course_json

def get_metadata_json(course_directory):
    course_dir = Path(course_directory)
    metadata_json = ""
    with (course_dir / 'unpacked/metadata/course_metadata.json').open('r') as f:
        metadata_json = json.loads(f.read())  
        
    return metadata_json

def get_spline_configuration_json(course_directory):
    try:
        course_dir = Path(course_directory)
        spline_json = None
        with (course_dir / 'splines.json').open('r') as f:
            spline_json = json.loads(f.read())  
            
        return spline_json
    except:
        return None

def write_course_json(course_directory, course_json):
    course_dir = Path(course_directory)
    with (course_dir / 'unpacked/course_description/course_description.json').open('w') as f:
        # Reduce floating point resolution to save file space.  Round to millimeter
        # Workaround since dumps has no precision
        # https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
        f.write(json.dumps(json.loads(json.dumps(course_json), parse_float=lambda x: round(float(x), 3)), separators=(',', ':')))

def write_metadata_json(course_directory, metadata_json):
    course_dir = Path(course_directory)
    with (course_dir / 'unpacked/metadata/course_metadata.json').open('w') as f:
        out = json.dumps(metadata_json, separators=(',', ':'))
        f.write(out)

def set_course_metadata_name(course_directory, new_course_name):
    metadata_json = get_metadata_json(course_directory)
    metadata_json["name"] = new_course_name
    write_metadata_json(course_directory, metadata_json)

def waypoint_dist(p1, p2):
    dx = p1["x"] - p2["x"]
    dz = p1["z"] - p2["z"]
    return math.sqrt(dx**2 + dz**2)



def get_hole_information(course_json, course_version):
    pars = []
    pin_counts = []
    tees = [[], [], [], [], []]

    if course_version not in tgc_definitions.version_tags:
        return

    hole_tag = tgc_definitions.version_tags[course_version]['holes']
    pin_tag  = tgc_definitions.version_tags[course_version]['pins']
    tee_tag  = tgc_definitions.version_tags[course_version]['tees']

    for h in course_json.get(hole_tag, []): # Avoid key error if hole_tag is missing
        # Par: creatorDefinedPar overrides; fall back to par; default 0
        par = h.get("creatorDefinedPar", 0)
        if par <= 0:
            par = h.get("par", 0)
        pars.append(par)

        pin_counts.append(len(h.get(pin_tag, []))) # Fallback to empty list

        # Waypoints
        wps = h.get("waypoints", []) # Fallback to empty list
        waypoints = wps[1:]                # every point but the first
        start_wp = wps[0] if wps else None # may be missing on empty templates

        # Common distance along fairway
        common_distance = 0.0
        for i in range(0, len(waypoints) - 1):
            common_distance += waypoint_dist(waypoints[i], waypoints[i + 1])

        # Tees (up to 5). If no waypoint[0], yardage is unknown -> None.
        for i in range(0, 5):
            tee_list = h.get(tee_tag, []) # Fallback to empty list
            if i < len(tee_list) and start_wp is not None:
                tp = tee_list[i]
                if course_version == 25:
                    tp = tp.get("position", tp)
                total_dist = common_distance + waypoint_dist(tp, start_wp)
                tees[i].append(1.09 * total_dist)  # meters -> yards approx
            else:
                tees[i].append(None)

    return pars, pin_counts, tees


def strip_terrain(course_json, output_file, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.setdefault("userLayers2", {})
    else:
        layer_json = course_json.setdefault("userLayers", {})

    # Copy existing terrain and write to disk
    output_data = {
        'terrainHeight': layer_json.get("terrainHeight", []),
        'height':        layer_json.get("height", [])
    }

    print("Saving Terrain as " + output_file)
    np.save(output_file, output_data)

    # Clear existing terrain (safe even if keys were missing)
    layer_json["terrainHeight"] = []
    layer_json["height"] = []

    return course_json

def insert_terrain(course_json, input_file, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.setdefault("userLayers2", {})
    else:
        layer_json = course_json.setdefault("userLayers", {})

    print("Loading terrain from: " + input_file)
    read_dictionary = np.load(input_file, allow_pickle=True).item()

    # Copy terrain in (use .get(...) on the file dict to avoid KeyError)
    layer_json["terrainHeight"] = read_dictionary.get("terrainHeight", [])
    layer_json["height"]        = read_dictionary.get("height", [])

    return course_json

def strip_holes(course_json, output_file, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    hole_tag = tgc_definitions.version_tags[course_version]['holes']

    # Copy existing holes and write to disk
    output_data = {'holes': course_json.get(hole_tag, [])}

    print("Saving Holes as " + output_file)
    np.save(output_file, output_data)

    # Clear existing holes (ensure key exists and attaches to course_json)
    course_json.setdefault(hole_tag, [])
    course_json[hole_tag] = []

    return course_json

def insert_holes(course_json, input_file, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    hole_tag = tgc_definitions.version_tags[course_version]['holes']

    print("Loading holes from: " + input_file)
    read_dictionary = np.load(input_file, allow_pickle=True).item()

    # Replace our holes from those in the file (avoid KeyError on file contents)
    course_json[hole_tag] = read_dictionary.get('holes', [])

    return course_json

# Shift terrain and features are separate in case they need to be lined up with each other
def shift_terrain(course_json, easting_shift, northing_shift, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})

    for i in layer_json.get("height", []):
        pos = i.get('position')
        if pos:
            if 'x' in pos: pos['x'] += easting_shift
            if 'z' in pos: pos['z'] += northing_shift

    for i in layer_json.get("terrainHeight", []):
        pos = i.get('position')
        if pos:
            if 'x' in pos: pos['x'] += easting_shift
            if 'z' in pos: pos['z'] += northing_shift

    return course_json

def shift_features(course_json, easting_shift, northing_shift, course_version):
    """Shift feature geometry defensively across schema variants.

    Some 2K25 templates omit legacy keys like 'surfaceBrushes' or nest brushes
    under userLayers/layers. Use .get(...) and type checks to avoid KeyErrors.
    """
    if course_version not in tgc_definitions.version_tags:
        return

    dim2 = 'y'
    if course_version == 25:
        layer_json = course_json
        dim2 = 'z'
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})
    
    hole_tag   = tgc_definitions.version_tags[course_version]['holes']
    tee_tag    = tgc_definitions.version_tags[course_version]['tees']
    crowd_tag  = tgc_definitions.version_tags[course_version]['crowd']
    spline_tag = tgc_definitions.version_tags[course_version]['splines']
    surface_tag= tgc_definitions.version_tags[course_version]['surfaces']
    oob_tag    = tgc_definitions.version_tags[course_version]['oob']
    obj_tag    = tgc_definitions.version_tags[course_version]['objects']

    # Shift splines (waypoints)
    for i in (course_json.get(spline_tag, []) or layer_json.get(spline_tag, [])): # tag may live at root or under layer_json
        for wp in i.get("waypoints", []):
            if "pointOne" in wp and "x" in wp["pointOne"]:
                wp["pointOne"]["x"] += easting_shift
            if "pointTwo" in wp and "x" in wp["pointTwo"]:
                wp["pointTwo"]["x"] += easting_shift
            if "waypoint" in wp and "x" in wp["waypoint"]:
                wp["waypoint"]["x"] += easting_shift
            if "pointOne" in wp and dim2 in wp["pointOne"]:
                wp["pointOne"][dim2] += northing_shift
            if "pointTwo" in wp and dim2 in wp["pointTwo"]:
                wp["pointTwo"][dim2] += northing_shift
            if "waypoint" in wp and dim2 in wp["waypoint"]:
                wp["waypoint"][dim2] += northing_shift

    # Shift holes (waypoints + tee positions)
    for h in course_json.get(hole_tag, []):
        for w in h.get("waypoints", []):
            if "x" in w: w["x"] += easting_shift
            if "z" in w: w["z"] += northing_shift
        for t in h.get(tee_tag, []):
            tp = t["position"] if (course_version == 25 and "position" in t) else t
            if "x" in tp: tp["x"] += easting_shift
            if "z" in tp: tp["z"] += northing_shift
        # Pin positions are relative; no shift

    # Shift brushes (surfaces, water, oob, crowd)
    oob_json   = layer_json.get(oob_tag, [])
    crowd_json = layer_json.get(crowd_tag, [])
    if course_version == 25:
        if isinstance(oob_json, dict):
            oob_json = oob_json.get("brushes", [])
        if isinstance(crowd_json, dict):
            crowd_json = crowd_json.get("brushes", [])
    
    # Water can be list (legacy) or dict{"brushes":[...]} on 2K25
    water_json = layer_json.get("water", [])
    if course_version == 25 and isinstance(water_json, dict):
        water_json = water_json.get("brushes", [])

    for b in itertools.chain(layer_json.get(surface_tag, []),
                             water_json,
                             oob_json,
                             crowd_json):
        if 'position' in b:
            if 'x' in b['position']: b['position']['x'] += easting_shift
            if 'z' in b['position']: b['position']['z'] += northing_shift

    # Shift objects (items and clusters)
    for o in (course_json.get(obj_tag, []) or layer_json.get(obj_tag, [])):
        val = o.get("Value", {})
        for i in val.get("items", []):
            if 'position' in i:
                if 'x' in i['position']: i['position']['x'] += easting_shift
                if 'z' in i['position']: i['position']['z'] += northing_shift
        for c in val.get("clusters", []):
            if 'position' in c:
                if 'x' in c['position']: c['position']['x'] += easting_shift
                if 'z' in c['position']: c['position']['z'] += northing_shift

    return course_json


def shift_course(course_json, easting_shift, northing_shift, course_version):
    course_json = shift_terrain(course_json, easting_shift, northing_shift, course_version)
    return shift_features(course_json, easting_shift, northing_shift, course_version)

# Helper function to rotate coordinates on many different element types
def rotateCoord(elem, x_key='x', y_key='y', c=1.0, s=0.0):
    x = elem[x_key]
    y = elem[y_key]
    elem[x_key] = x * c - y * s
    elem[y_key] = x * s + y * c

# Rotation angle is positive around the y-DOWN axis
# Positive values will rotate the course clockwise
def rotate_course(course_json, rotation_angle_radians, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    dim2 = 'y'
    if course_version == 25:
        layer_json = course_json
        dim2 = 'z'
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})
    
    hole_tag   = tgc_definitions.version_tags[course_version]['holes']
    tee_tag    = tgc_definitions.version_tags[course_version]['tees']
    pin_tag    = tgc_definitions.version_tags[course_version]['pins']
    crowd_tag  = tgc_definitions.version_tags[course_version]['crowd']    
    spline_tag = tgc_definitions.version_tags[course_version]['splines']
    surface_tag= tgc_definitions.version_tags[course_version]['surfaces']
    oob_tag    = tgc_definitions.version_tags[course_version]['oob']
    obj_tag    = tgc_definitions.version_tags[course_version]['objects']

    # Elements that have rotation values are stored in degrees
    rotation_angle_degrees = 180.0 * rotation_angle_radians / math.pi

    # Pre calculate cosine and sine with what would be the y-up angle
    c = math.cos(-rotation_angle_radians)
    s = math.sin(-rotation_angle_radians)

    # Unwrap lists that might be dict{"brushes":[...]} in 2K25
    surfaces_json = layer_json.get(surface_tag, [])
    if course_version == 25 and isinstance(surfaces_json, dict):
        surfaces_json = surfaces_json.get("brushes", [])

    water_json = layer_json.get("water", [])
    if course_version == 25 and isinstance(water_json, dict):
        water_json = water_json.get("brushes", [])

    oob_json = layer_json.get(oob_tag, [])
    if course_version == 25 and isinstance(oob_json, dict):
        oob_json = oob_json.get("brushes", [])

    crowd_json = layer_json.get(crowd_tag, [])
    if course_version == 25 and isinstance(crowd_json, dict):
        crowd_json = crowd_json.get("brushes", [])

    # Rotate Brushes (height/terrainHeight/surfaces/water/oob/crowd)
    for b in itertools.chain(layer_json.get("height", []),
                             layer_json.get("terrainHeight", []),
                             surfaces_json,
                             water_json,
                             oob_json,
                             crowd_json):
        pos = b.get('position')
        if pos:
            try:
                rotateCoord(pos, 'x', 'z', c, s)
            except Exception:
                pass
        rot = b.get('rotation')
        if rot and 'y' in rot:
            rot['y'] += rotation_angle_degrees

    # Rotate splines (tag may live at root or under layer_json)
    for i in (course_json.get(spline_tag, []) or layer_json.get(spline_tag, [])):
        for wp in i.get("waypoints", []):
            if "pointOne" in wp: rotateCoord(wp["pointOne"], 'x', dim2, c, s)
            if "pointTwo" in wp: rotateCoord(wp["pointTwo"], 'x', dim2, c, s)
            if "waypoint" in wp: rotateCoord(wp["waypoint"], 'x', dim2, c, s)

    # Rotate Holes
    pin_dim2 = 'z' if course_version == 25 else 'y'
    for h in course_json.get(hole_tag, []):
        for w in h.get("waypoints", []):
            try:
                rotateCoord(w, 'x', 'z', c, s)
            except Exception:
                pass
        for t in h.get(tee_tag, []):
            tp = t.get("position", t) if course_version == 25 else t
            try:
                rotateCoord(tp, 'x', 'z', c, s)
            except Exception:
                pass
        for p in h.get(pin_tag, []):
            pp = p.get("position", p) if course_version == 25 else p
            try:
                rotateCoord(pp, 'x', pin_dim2, c, s)
            except Exception:
                pass

    # Rotate Objects (root or layer)
    for o in (course_json.get(obj_tag, []) or layer_json.get(obj_tag, [])):
        val = o.get("Value", {})
        for i in val.get("items", []):
            pos = i.get("position")
            if pos:
                try:
                    rotateCoord(pos, 'x', 'z', c, s)
                except Exception:
                    pass
            rot = i.get('rotation')
            if rot and 'y' in rot:
                rot['y'] += rotation_angle_degrees

        for cl in val.get("clusters", []):
            pos = cl.get("position")
            if pos:
                try:
                    rotateCoord(pos, 'x', 'z', c, s)
                except Exception:
                    pass
            rot = cl.get('rotation')
            if rot and 'y' in rot:
                rot['y'] += rotation_angle_degrees

    return course_json

def getCoursePoints(course_json, course_version):
    if course_version not in tgc_definitions.version_tags:
        return []

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})

    cv2_pts = []

    for i in layer_json.get("height", []):
        pos = i.get("position")
        if pos and 'x' in pos and 'z' in pos:
            cv2_pts.append([pos['x'], pos['z']])

    for i in layer_json.get("terrainHeight", []):
        pos = i.get("position")
        if pos and 'x' in pos and 'z' in pos:
            cv2_pts.append([pos['x'], pos['z']])

    return cv2_pts


def getBoundingBox(course_json, course_version):
    cv2_pts = getCoursePoints(course_json, course_version)
    if not cv2_pts:
        return (0, 0, 0, 0)  # safe default for empty courses/templates
    arr = np.array(cv2_pts, dtype=np.int32)
    return cv2.boundingRect(arr)


def getMinBoundingBox(course_json, course_version):
    cv2_pts = getCoursePoints(course_json, course_version)
    if not cv2_pts:
        return ((0, 0), (0, 0), 0.0)  # safe default
    arr = np.array(cv2_pts, dtype=np.int32)
    return cv2.minAreaRect(arr)

def setValues(x, y, ll, ul, ur, lr):
    # r^2 > dist^2, so no need to do square root
    r2 = x**2 + y**2
    if x <= 0.0 and y <= 0.0:
        pdist2 = ll[0]**2 + ll[1]**2
        if r2 > pdist2:
            ll = (x, y, r2)
            return (ll, ul, ur, lr)
    elif x <= 0.0 and y >= 0.0:
        pdist2 = ul[0]**2 + ul[1]**2
        if r2 > pdist2:
            ul = (x, y, r2)
            return (ll, ul, ur, lr)
    elif x >= 0.0 and y >= 0.0:
        pdist2 = ur[0]**2 + ur[1]**2
        if r2 > pdist2:
            ur = (x, y, r2)
            return (ll, ul, ur, lr)
    elif x >= 0.0 and y <= 0.0:
        pdist2 = lr[0]**2 + lr[1]**2
        if r2 > pdist2:
            lr = (x, y, r2)
            return (ll, ul, ur, lr)
    return (ll, ul, ur, lr)

# Assuming terrain always goes further than "other stuff"
# Also assumes course is roughly centered at 0,0
# Returns ll, ul, ur, lr
def get_terrain_extremes(course_json, course_version):
    if course_version not in tgc_definitions.version_tags:
        return

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.get("userLayers2", {})
    else:
        layer_json = course_json.get("userLayers", {})

    # X, Z, radius_squared to point
    ll = (0.0, 0.0, 0.0)
    ul = (0.0, 0.0, 0.0)
    ur = (0.0, 0.0, 0.0)
    lr = (0.0, 0.0, 0.0)

    for i in layer_json.get("height", []):
        pos = i.get("position")
        if pos and "x" in pos and "z" in pos:
            ll, ul, ur, lr = setValues(pos["x"], pos["z"], ll, ul, ur, lr)

    for i in layer_json.get("terrainHeight", []):
        pos = i.get("position")
        if pos and "x" in pos and "z" in pos:
            ll, ul, ur, lr = setValues(pos["x"], pos["z"], ll, ul, ur, lr)

    return (ll, ul, ur, lr)

# Determines the four extremes and tries to shift and rotate the course to fit within 2000m
def auto_position_course(course_json, printf=print, course_version=-1):
    if course_version not in tgc_definitions.version_tags:
        return course_json  # keep behavior consistent with other helpers

    extremes = get_terrain_extremes(course_json, course_version)
    if extremes is None:
        extremes = ((0.0, 0.0, 0.0),) * 4  # safe default

    rect = getBoundingBox(course_json, course_version)

    fits_on_map = (
        -1000.0 <= rect[0] and
        -1000.0 <= rect[1] and
        rect[0] + rect[2] <= 1000.0 and
        rect[1] + rect[3] <= 1000.0
    )
    if fits_on_map:
        printf("Course fits within map")

    # If course would fit within 2000x2000, don't try to rotate it
    rotation = 0.0
    if rect[2] > 2000.0 or rect[3] > 2000.0:
        ideal_angles = [-3.0/4.0*math.pi, 3.0/4.0*math.pi, 1.0/4.0*math.pi, -1.0/4.0*math.pi]
        rotation_sum = 0.0
        for c, a in zip(extremes, ideal_angles):
            angle = math.atan2(c[1], c[0])
            angle_diff = abs(a - angle)
            rotation_sum += abs(angle_diff)

        rotation = rotation_sum / float(len(extremes))
        printf("Rotating course by: " + str(rotation))
        course_json = rotate_course(course_json, rotation, course_version)

        # See if opposite rotation is better
        rect = getBoundingBox(course_json, course_version)
        if rect[2] > 2000.0 or rect[3] > 2000.0:
            printf("Trying opposite rotation: " + str(-rotation))
            course_json = rotate_course(course_json, -2.0 * rotation, course_version)

    # Final translation to center
    rect = getBoundingBox(course_json, course_version)
    eastwest_shift   = -rect[2] / 2 - rect[0]
    northsouth_shift = -rect[3] / 2 - rect[1]
    printf("Shift course by: " + str(eastwest_shift) + " x " + str(northsouth_shift))

    return shift_course(course_json, eastwest_shift, northsouth_shift, course_version)

# This doesn't work perfectly, but it works for many courses
def auto_merge_courses(course1_json, course2_json, course_version):
    # Shift and rotate courses so that they don't overlap
    # Get bounding boxes for each course
    bb1 = getMinBoundingBox(course1_json, course_version)
    bb2 = getMinBoundingBox(course2_json, course_version)

    # Find which course is larger
    larger_course = course1_json
    smaller_course = course2_json
    if bb2[1][0] + bb2[1][1] > bb1[1][0] + bb1[1][1]:
        larger_course = course2_json
        smaller_course = course1_json

    # Fit the larger course section on any way we can
    larger_course = auto_position_course(larger_course, course_version=course_version)

    # Find enough space for the smaller course on the map
    bb1 = getMinBoundingBox(larger_course, course_version)
    bb2 = getMinBoundingBox(smaller_course, course_version)

    # Rotate smaller course to match larger course
    larger_horizontal_aligned  = bb1[1][1] > bb1[1][0]
    smaller_horizontal_aligned = bb2[1][1] > bb2[1][0]
    rotation_angle = (bb1[2] - bb2[2]) * math.pi / 180.0
    if larger_horizontal_aligned == smaller_horizontal_aligned:
        rotation_angle += math.pi / 2.0  # Rotate an extra 90 to align major distance
    rotate_course(smaller_course, rotation_angle, course_version)

    # Shift courses to not overlap
    bb1 = getMinBoundingBox(larger_course, course_version)
    bb2 = getMinBoundingBox(smaller_course, course_version)

    # Determine dominant angle for larger course
    # Needed because minAreaRect only reports -90 to 0.0
    rotation = 0.0
    radius   = 0.0  # (h0/2 + gap + h1/2)
    orig_angle = math.pi / 180.0 * bb1[2]
    if bb1[1][0] > bb1[1][1]:  # height > width
        radius = (bb1[1][0] / 2.0 + 20 + bb2[1][0] / 2.0)
        if orig_angle < -75:
            rotation = 0.0                 # Shift straight right
        else:
            rotation = math.pi / 4.0       # Shift at 45 degrees to upper right
    else:
        radius = (bb1[1][1] / 2.0 + 20 + bb2[1][1] / 2.0)
        if orig_angle < -75:
            rotation = math.pi / 2.0       # Shift up
        else:
            rotation = 3.0 * math.pi / 4.0 # Shift at 45 degrees to upper left

    # x0 + radius * sin/cos(rotation)
    # Angles are -90 to 0.0, invert so courses shift up or to the right
    new_center_x = bb1[0][0] + radius * math.cos(rotation)
    new_center_y = bb1[0][1] + radius * math.sin(rotation)
    offset_x = new_center_x - bb2[0][0]
    offset_y = new_center_y - bb2[0][1]
    smaller_course = shift_course(smaller_course, offset_x, offset_y, course_version)

    # Apply usual merge
    merged_course = merge_courses(course1_json, course2_json, course_version=course_version)  # keep call as-is

    # Position this combined course as best as possible
    return auto_position_course(merged_course, course_version=course_version)

def merge_courses(course1_json, course2_json, course_version):
    if course_version not in tgc_definitions.version_tags:
        return course1_json

    # Resolve layer containers
    if course_version == 25:
        layer1 = course1_json
        layer2 = course2_json
    elif course_version == 23:
        layer1 = course1_json.setdefault("userLayers2", {})
        layer2 = course2_json.get("userLayers2", {})
    else:
        layer1 = course1_json.setdefault("userLayers", {})
        layer2 = course2_json.get("userLayers", {})

    tags = tgc_definitions.version_tags[course_version]
    hole_tag    = tags['holes']
    spline_tag  = tags['splines']
    surface_tag = tags['surfaces']
    oob_tag     = tags['oob']
    crowd_tag   = tags['crowd']
    obj_tag     = tags['objects']

    # --- Terrain ---
    layer1.setdefault("height", []).extend(layer2.get("height", []))
    layer1.setdefault("terrainHeight", []).extend(layer2.get("terrainHeight", []))

    # --- Surfaces/Water/OOB/Crowd (2K25 may wrap as {"brushes":[...]}) ---
    def _src_brushes(layer, key):
        v = layer.get(key, [])
        if course_version == 25 and isinstance(v, dict):
            v = v.get("brushes", [])
        return v

    def _dst_brush_list(layer, key):
        # Ensure destination is a list; if 2K25 provides a dict, ensure/return its .brushes list.
        v = layer.setdefault(key, [])
        if course_version == 25 and isinstance(v, dict):
            v = v.setdefault("brushes", [])
        return v

    _dst_brush_list(layer1, surface_tag).extend(_src_brushes(layer2, surface_tag))
    _dst_brush_list(layer1, "water").extend(_src_brushes(layer2, "water"))
    _dst_brush_list(layer1, oob_tag).extend(_src_brushes(layer2, oob_tag))
    _dst_brush_list(layer1, crowd_tag).extend(_src_brushes(layer2, crowd_tag))

    # --- Splines (may live at root or under layer) ---
    src_splines = (course2_json.get(spline_tag, []) or layer2.get(spline_tag, []))
    if spline_tag in course1_json:
        dst_splines = course1_json[spline_tag]
    else:
        dst_splines = layer1.setdefault(spline_tag, [])
    dst_splines.extend(src_splines)

    # --- Objects (may live at root or under layer) ---
    src_objs = (course2_json.get(obj_tag, []) or layer2.get(obj_tag, []))
    if obj_tag in course1_json:
        dst_objs = course1_json[obj_tag]
    else:
        dst_objs = layer1.setdefault(obj_tag, [])
    dst_objs.extend(src_objs)

    # --- Holes (root) ---
    print("Warning, holes may be out of order")
    course1_json.setdefault(hole_tag, [])
    for i in course2_json.get(hole_tag, []):
        if len(course1_json[hole_tag]) < 18:
            course1_json[hole_tag].append(i)
        else:
            print("Too many holes")

    return course1_json

def elevate_terrain(course_json, elevate_shift, buffer_height=10.0, clip_lowest_value=-2.0, printf=print, course_version=-1):
    if course_version not in tgc_definitions.version_tags:
        printf("invalid version")
        printf(course_version)
        return None

    # Pick the layer container (attach when we will write)
    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json.setdefault("userLayers2", {})
    else:
        layer_json = course_json.setdefault("userLayers", {})

    # Automatic terrain shift
    if elevate_shift == 0.0 or elevate_shift is None:
        elevations = []
        for i in layer_json.get("height", []):
            v = i.get('value')
            if isinstance(v, (int, float)):
                elevations.append(v)
        for i in layer_json.get("terrainHeight", []):
            v = i.get('value')
            if isinstance(v, (int, float)):
                elevations.append(v)

        if not elevations or len(elevations) < 2:
            printf("Course likely does not need elevation adjustment (insufficient terrain points)")
            return course_json

        elevations = np.array(elevations)
        s = np.sort(elevations)
        g = np.gradient(s)

        half_length = round(len(elevations) / 2)
        if half_length == 0:
            printf("Course likely does not need elevation adjustment")
            return course_json

        diff_threshold = (np.median(g[0:half_length]) + g[0:half_length].max(axis=0)) / 2.0
        diff_threshold = min(0.015, diff_threshold)  # cap threshold

        try:
            split_index = np.where(g[0:half_length] > diff_threshold)[0][-1] + 1
            elevate_shift = -min(s[split_index:]) + buffer_height
        except IndexError:
            printf("Course likely does not need elevation adjustment")
            return course_json

    printf("Shifting elevation by: " + str(elevate_shift))

    # Apply shift and clip low values
    remaining_height = []
    for i in layer_json.get("height", []):
        v = i.get('value')
        if isinstance(v, (int, float)) and v + elevate_shift >= clip_lowest_value:
            i['value'] = v + elevate_shift
            remaining_height.append(i)
    layer_json["height"] = remaining_height

    remaining_terrain = []
    for i in layer_json.get("terrainHeight", []):
        v = i.get('value')
        if isinstance(v, (int, float)) and v + elevate_shift >= clip_lowest_value:
            i['value'] = v + elevate_shift
            remaining_terrain.append(i)
    layer_json["terrainHeight"] = remaining_terrain

    return course_json

# Maximum course size is 2000 meters by 2000 meters.
# This crops if anything is further than max from the origin
# 2000.0 / 2 is 1000.0 meters
def crop_course(course_json, max_easting=1000.0, max_northing=1000.0, course_version=-1):
    if course_version not in tgc_definitions.version_tags:
        print("invalid version")
        print(course_version)
        return None

    # layer container (attach if we write)
    if course_version == 25:
        layer_json = course_json
        dim2 = 'z'
    elif course_version == 23:
        layer_json = course_json.setdefault("userLayers2", {})
        dim2 = 'y'
    else:
        layer_json = course_json.setdefault("userLayers", {})
        dim2 = 'y'
         
    spline_tag = tgc_definitions.version_tags[course_version]['splines']
    hole_tag   = tgc_definitions.version_tags[course_version]['holes']
    tee_tag    = tgc_definitions.version_tags[course_version]['tees']

    # Filter elevation
    remaining_height = []
    for i in layer_json.get("height", []):
        pos = i.get("position")
        if pos and abs(pos.get("x", 1e9)) <= max_easting and abs(pos.get("z", 1e9)) <= max_northing:
            remaining_height.append(i)
    layer_json["height"] = remaining_height

    remaining_terrain = []
    for i in layer_json.get("terrainHeight", []):
        pos = i.get("position")
        if pos and abs(pos.get("x", 1e9)) <= max_easting and abs(pos.get("z", 1e9)) <= max_northing:
            remaining_terrain.append(i)
    layer_json["terrainHeight"] = remaining_terrain

    # Filter splines (tag may live at root or under layer_json)
    remaining_splines = []
    for i in (course_json.get(spline_tag, []) or layer_json.get(spline_tag, [])):
        keep_spline = True
        for wp in i.get("waypoints", []):
            p1 = wp.get("pointOne", {})
            p2 = wp.get("pointTwo", {})
            p  = wp.get("waypoint", {})
            if (abs(p1.get("x", 1e9)) <= max_easting and
                abs(p2.get("x", 1e9)) <= max_easting and
                abs(p.get("x", 1e9))  <= max_easting and
                abs(p1.get(dim2, 1e9)) <= max_northing and
                abs(p2.get(dim2, 1e9)) <= max_northing and
                abs(p.get(dim2, 1e9))  <= max_northing):
                continue
            else:
                keep_spline = False
                break
        if keep_spline:
            remaining_splines.append(i)
    if spline_tag in course_json:
        course_json[spline_tag] = remaining_splines
    else:
        layer_json[spline_tag] = remaining_splines

    # Filter Holes
    remaining_holes = []
    for h in course_json.get(hole_tag, []):
        keep_hole = True
        for w in h.get("waypoints", []):
            if abs(w.get("x", 1e9)) <= max_easting and abs(w.get("z", 1e9)) <= max_northing:
                continue
            else:
                keep_hole = False
                break
 
        if keep_hole:
            for t in h.get(tee_tag, []):
                tp = t.get("position", t) if course_version == 25 else t
                if abs(tp.get("x", 1e9)) <= max_easting and abs(tp.get("z", 1e9)) <= max_northing:
                    continue
                else:
                    keep_hole = False
                    break

        if keep_hole:
            remaining_holes.append(h)
    course_json[hole_tag] = remaining_holes

    return course_json
