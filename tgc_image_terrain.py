import cv2
import itertools
import json
import math
import numpy as np
from pathlib import Path
import random
import sys
import time
import traceback

from GeoPointCloud import GeoPointCloud
from infill_image import infill_image_scipy
import OSMTGC
import tgc_definitions
import tgc_tools

from anti_ripple import anti_ripple_bilateral, ripple_metric_safe, ripple_diag # ripple_metric, ripple_metric_debug, 


status_print_duration = 1.0 # Print progress every n seconds

def get_pixel(x_pos, z_pos, height, scale, brush_type=72):
    output = json.loads('{"tool":0,"position":{"x":0.0,"y":"-Infinity","z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0},"_orientation":0.0,"scale":{"x":1.0, \
                         "y":1.0,"z":1.0},"type":0,"value":0.0,"holeId":-1,"radius":0.0,"orientation":0.0}')
    output['type'] = brush_type
    output['position']['x'] = x_pos
    output['position']['z'] = z_pos
    output['value'] = height
    output['scale']['x'] = scale
    output['scale']['z'] = scale
    return output

def get_object_item(x_pos, z_pos, rotation_degrees):
    output = json.loads('{"position":{"x":0.0,"y":"-Infinity","z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}')
    output['position']['x'] = x_pos
    output['position']['z'] = z_pos
    output['rotation']['y'] = rotation_degrees
    return output

def get_placed_object(course_version=-1):
    key = '{"category":0,"type":0,"theme":true}'
    if course_version >= 23:
        key = '{"path":""}'

    output = json.loads('{"Key":'+key+',"Value":{"items":[],"clusters":[]}}')
    
    return output

def get_trees(theme, tree_variety, trees, course_version=-1):
    # Get possible trees for this theme.  User can't easily change theme after this
    # But it's easy to rerun the import tool
    # Default to the default tree 0 if empty or not found
    if course_version >= 23:
        normal_tree_ids = tgc_definitions.normal_trees_2k.get(theme, [0])
    else:
        normal_tree_ids = tgc_definitions.normal_trees.get(theme, [0])

    if (not tree_variety) or len(normal_tree_ids) == 0:
        normal_tree_ids = [0]
    # Default to the normal trees if empty or not found
    if course_version >= 23:
        skinny_tree_ids = tgc_definitions.skinny_trees_2k.get(theme, normal_tree_ids)
    else:
        skinny_tree_ids = tgc_definitions.skinny_trees.get(theme, normal_tree_ids)

    if (not tree_variety) or len(skinny_tree_ids) == 0:
        skinny_tree_ids = []

    # Make an group for each type of tree, even if they may not be used
    normal_trees = []
    for tree_id in normal_tree_ids:
        p = get_placed_object(course_version)
        if course_version >= 23:
            p['Key']['path'] = tgc_definitions.trees_2k[tree_id]
        else:
            p['Key']['category'] = 0
            p['Key']['type'] = tree_id
        normal_trees.append(p)
    skinny_trees = []
    for tree_id in skinny_tree_ids:
        p = get_placed_object(course_version)
        if course_version >= 23:
            p['Key']['path'] = tgc_definitions.trees_2k[tree_id]
        else:
            p['Key']['category'] = 0
            p['Key']['type'] = tree_id
        skinny_trees.append(p)

    # Scale trees based on relative sizes
    min_radius_scale = 0.2
    radius_scale_range = 1.5 - min_radius_scale
    min_height_scale = 0.5
    height_scale_range = 1.2 - min_height_scale

    min_tree_radius = min(trees, key=lambda x: x[2])[2]
    max_tree_radius = max(trees, key=lambda x: x[2])[2]
    tree_radius_range = max_tree_radius - min_tree_radius
    if tree_radius_range > 0.01:
        radius_multiplier = radius_scale_range / tree_radius_range
    else:
        # All nearly same radius, scale to 1.0
        min_radius_scale = 1.0
        radius_multiplier = 0.0
    min_tree_height = min(trees, key=lambda x: x[3])[3]
    max_tree_height = max(trees, key=lambda x: x[3])[3]
    tree_height_range = max_tree_height - min_tree_height
    if tree_height_range > 0.01:
        height_multiplier = height_scale_range / tree_height_range
    else:
        # All nearly same height, scale to 1.0
        min_height_scale = 1.0
        height_multiplier = 0.0

    for tree in trees:
        easting, northing, r, h = tree
        t = get_object_item(easting, northing, random.randrange(0, 359))

        t['scale']['y'] = (h-min_tree_height)*height_multiplier + min_height_scale
        t['scale']['x'] = (r-min_tree_radius)*radius_multiplier + min_radius_scale
        t['scale']['z'] = (r-min_tree_radius)*radius_multiplier + min_radius_scale

        if h / r < 2.5 or len(skinny_trees) == 0: # Normal Tree
            group = random.choice(normal_trees)
        else: # Skinny tree
            group = random.choice(skinny_trees)
        group['Value']['items'].append(t)

    # Remove empty groups
    output = []
    for g in itertools.chain(normal_trees, skinny_trees):
        if len(g['Value']['items']) > 0:
            output.append(g)
    return output

def get_lidar_trees(theme, tree_variety, lidar_trees, pc, mask, mask_pc, image_scale, course_version=-1):
    # Convert to TGC coordinates
    trees = []
    for tree in lidar_trees:
        easting, northing, r, h = tree
        # Use mask to only add trees on desired areas
        row, column = mask_pc.projToCV2(easting, northing, image_scale)
        mask_color = mask[(row, column)]
        # Color order is BGR, support both MS Paint Red Colors
        if not (mask_color[0] < 40 and mask_color[1] < 40 and mask_color[2] > 130):
            # Use standard pointcloud tp project trees into final TGC coordinates
            x, y, z = pc.projToTGC(easting, northing, 0.0)
            trees.append((x, z, r, h))

    return get_trees(theme, tree_variety, trees, course_version)

# Set various constants that we need
def set_constants(course_json, flatten_fairways=False, flatten_greens=False, course_latitude=None, printf=print):
    # These only work if the terrain is made from scult (red brushes) rather than landscape (blue brushes)
    course_json["flattenFairways"] = flatten_fairways # Needed to not flatten under fairway splines
    course_json["flattenGreens"] = flatten_greens # Needed to not flatten under green splines

    # Slow down and soften greens a bit by default.   A lot of real world courses are not as firm as fast as TGC defaults
    course_json["greenSpeed"] = 0.136312455
    course_json["greenFirmness"] = 0.20308432

    # Set all hole sizes to 0.0 so that you can add holes without the autogenerated features appearing
    # This lets a designer manually specifcy or delete and readd a hole without the teebox, green, bunkers appearing over their OSM.
    course_json["roughRadius"] = 0.0
    course_json["heavyRoughRadius"] = 0.0
    course_json["fairwayRadius"] = 0.0
    course_json["greenRadius"] = 0.0
    course_json["teeRadius"] = 0.0
    course_json["hazardGreenCount"] = 0.0
    course_json["hazardFairwayCount"] = 0.0

    # Set lighting parameters based on the actual position of the course
    course_json["sunOrientation"] = 0.0 # Set so in game North aligns to true North
    course_json["sunInclination"] = 45.0 # Default halfway position
    # Try to set inclination based on latitude, if provided
    # Based on: http://www.physicalgeography.net/fundamentals/6h.html
    if course_latitude is not None:
        earth_declination = 23.5
        if course_latitude >= earth_declination: # Above the Tropic of Cancer, use June Solstice position
            course_json["sunInclination"] = 90.0 - float(course_latitude) + earth_declination
        elif course_latitude >= 0.0: # Equation to the Tropic of Cancer, use Equinox position
            course_json["sunInclination"] = 90.0 - float(course_latitude)
        elif course_latitude >= -1.0*earth_declination: # Equation to the Tropic of Capricorn, use Equinox position
            # You can't set above 90 in game, but we can here.  Do this so that North and Sunrise to the East are still correct.
            course_json["sunInclination"] = 90.0 - float(course_latitude)
        else: # Below the Tropic of Caprion.  Use December Solstice position
            # You can't set above 90 in game, but we can here.  Do this so that North and Sunrise to the East are still correct.
            course_json["sunInclination"] = 90.0 + -1.0*float(course_latitude) - earth_declination
        printf("For latitude " + str(course_latitude) + ": Setting sun angle to: " + str(course_json["sunInclination"]))

    # Add our own JSON element so the courses could be filtered easily
    # Am choosing an organization name so that TGC-Designer-Tools could be forked
    course_json["gis"] = "ChadRockeyDevelopment"

    return course_json

def generate_course(course_json, heightmap_dir_path, options_dict={}, printf=print, course_version=-1):
    if course_version not in tgc_definitions.version_tags:
        print("invalid version")
        print(course_version)
        return None

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json["userLayers2"]
    else:
        layer_json = course_json["userLayers"]

    obj_tag = tgc_definitions.version_tags[course_version]['objects']
    printf("Loading data from " + heightmap_dir_path)

    # Infill data to prevent holes and make the data nice and smooth
    hm_file = Path(heightmap_dir_path) / '/heightmap.npy'
    try:
        read_dictionary = np.load(heightmap_dir_path + '/heightmap.npy', allow_pickle=True).item()
        im = read_dictionary['heightmap'].astype('float32')

        mask = cv2.imread(heightmap_dir_path + '/mask.png', cv2.IMREAD_COLOR)
        # Turn mask into matrix order from image order
        mask = np.flip(mask, 0)

        # Process Image
        printf("Filling holes in heightmap")
        image_scale = read_dictionary['image_scale']
        printf("Map scale is: " + str(image_scale) + " meters")
        background_ratio = None
        if options_dict.get('add_background', False):
            background_scale = float(options_dict.get('background_scale', 16.0))
            background_ratio = background_scale/image_scale
            printf("Background requested with scale: " + str(background_scale) + " meters")
            
        heightmap, background, holeMask = infill_image_scipy(im, mask, background_ratio=background_ratio, fill_water=options_dict.get('fill_water', False), purge_water=options_dict.get('purge_water', False), printf=printf)
    
    except FileNotFoundError:
        printf("Could not find heightmap or mask at: " + heightmap_dir_path)
        return course_json
    
    # --- Anti-ripple test pass  ---
    try:
        mpp = float(image_scale)  # meters per pixel for the high-res heightmap

        ar = options_dict.get("anti_ripple", {})
        enable = bool(ar.get("enabled", False))  # keep True while we test
        strength = float(ar.get("strength", 0.6))
        slope_thresh_deg = float(ar.get("slope_thresh_deg", 2.0))
        blend_width_deg = float(ar.get("blend_width_deg", 3.0))  # soft blend by default
        max_delta_m = float(ar.get("max_delta_m", 0.0))

        stamp_grid_m = float(ar.get("stamp_grid_m", mpp))  # default: match image_scale
        # optional guard if someone passes nonsense
        if stamp_grid_m <= 0:
            stamp_grid_m = mpp

        printf(f"Anti-ripple debug: shape={heightmap.shape}, mpp={mpp}, stamp={stamp_grid_m}, "
            f"strength={strength}, slope>= {slope_thresh_deg} deg, blend={blend_width_deg}, "
            f"clamp={max_delta_m} m")              

        if enable:

            #info = ripple_diag(heightmap, mpp)
            #printf(
            #    "Before AR diag: shape=%s mpp=%.3f slope[min=%.3f max=%.3f mean=%.3f] "
            #    "counts>=0.5:%d >=1:%d >=2:%d >=4:%d"
            #    % (info["shape"], info["mpp"], info["slope_min"], info["slope_max"], info["slope_mean"],
            #    info["n_ge_0_5"], info["n_ge_1"], info["n_ge_2"], info["n_ge_4"])
            #)                              

            before = ripple_metric_safe(heightmap, mpp, slope_min_deg=2.0)


            heightmap = anti_ripple_bilateral(
                heightmap,
                meters_per_px=mpp,
                stamp_grid_m=stamp_grid_m,
                strength=strength,
                slope_thresh_deg=slope_thresh_deg,
                blend_width_deg=blend_width_deg,
                max_delta_m=max_delta_m,
                clamp_window_px=(3, 31),
            )

            #info = ripple_diag(heightmap, mpp)
            #printf(
            #    "After AR diag: shape=%s mpp=%.3f slope[min=%.3f max=%.3f mean=%.3f] "
            #    "counts>=0.5:%d >=1:%d >=2:%d >=4:%d"
            #    % (info["shape"], info["mpp"], info["slope_min"], info["slope_max"], info["slope_mean"],
            #    info["n_ge_0_5"], info["n_ge_1"], info["n_ge_2"], info["n_ge_4"])
            #)                

            after = ripple_metric_safe(heightmap, mpp, slope_min_deg=2.0)

            printf(f"Anti-ripple: metric before={before:.4f} after={after:.4f}")

    except Exception as e:
        printf(f"Anti-ripple skipped: {e} | {traceback.format_exc()}")
    # ---------------------------------------------------    

    # Clear existing terrain
    course_json = set_constants(course_json, options_dict.get('flatten_fairways', False), options_dict.get('flatten_greens', False), read_dictionary['origin'][0], printf=printf)
    layer_json["height"] = []
    layer_json["terrainHeight"] = []
    course_json[obj_tag] = []

    # Construct high resolution model
    pc = GeoPointCloud()
    pc.addFromImage(heightmap, image_scale, read_dictionary['origin'], read_dictionary['projection'])

    # Add low resolution background
    if background is not None:
        background_pc = GeoPointCloud()
        background_pc.addFromImage(background, background_scale, read_dictionary['origin'], read_dictionary['projection'])
        num_points = len(background_pc.points())
        last_print_time = time.time()

        for n, i in enumerate(background_pc.points()):
            if time.time() > last_print_time + status_print_duration:
                last_print_time = time.time()
                printf(str(round(100.0*float(n) / num_points, 2)) + "% through heightmap")

            # Convert to projected coordinates, then project to TGC using the high resolution pointcloud to ensure alignment
            easting, northing = background_pc.enuToProj(i[0], i[1])
            x, y, z = pc.projToTGC(easting, northing, 0.0)
            # Using 10 - the very soft circles means we need to scale 2.5x more to fill and smooth the terrain
            layer_json["height"].append(get_pixel(x, z, i[2], 2.5*background_scale, brush_type=10))

    # Convert the pointcloud into height elements
    num_points = len(pc.points())
    last_print_time = time.time()
    for n, i in enumerate(pc.points()):
        if time.time() > last_print_time + status_print_duration:
            last_print_time = time.time()
            printf(str(round(100.0*float(n) / num_points, 2)) + "% through heightmap")

        x, y, z = pc.enuToTGC(i[0], i[1], 0.0) # Don't transform y, it's inverted from elevation

        smoothing = options_dict.get('smoothing', 0)
        if smoothing == 0:
            layer_json["height"].append(get_pixel(x, z, i[2], image_scale))
        elif smoothing == 1:
            layer_json["height"].append(get_pixel(x, z, i[2], 4*image_scale, brush_type=15))
        elif smoothing == 2:
            layer_json["height"].append(get_pixel(x, z, i[2], 3*image_scale, brush_type=9))
        elif smoothing == 3:
            layer_json["height"].append(get_pixel(x, z, i[2], 3*image_scale, brush_type=10))

    if options_dict.get('lidar_trees', False) and len(read_dictionary.get('trees', [])) > 0:
        printf("Adding trees from lidar data")
        # Need separate mask geopointcloud because pc is cropped
        mask_pc = GeoPointCloud()
        mask_pc.addFromImage(im, image_scale, read_dictionary['origin'], read_dictionary['projection'])
        for o in get_lidar_trees(course_json['theme'], options_dict.get('tree_variety', False), read_dictionary['trees'], pc, mask, mask_pc, image_scale, course_version):
            course_json[obj_tag].append(o)

    # Download OpenStreetMaps Data for this smaller area
    if options_dict.get('use_osm', True):
        printf("Adding golf features to lidar data")

        # Get spline configuration file, if present
        spline_json = tgc_tools.get_spline_configuration_json(heightmap_dir_path)

        # Use this data to create playable courses automatically
        upper_left_enu = pc.ulENU()
        lower_right_enu = pc.lrENU()
        upper_left_latlon = pc.enuToLatLon(*upper_left_enu)
        lower_right_latlon = pc.enuToLatLon(*lower_right_enu)
        # Order is South, West, North, East
        result = OSMTGC.getOSMData(lower_right_latlon[0], upper_left_latlon[1], upper_left_latlon[0], lower_right_latlon[1], printf=printf)
        osm_trees = OSMTGC.addOSMToTGC(course_json, pc, result, x_offset=float(options_dict.get('adjust_ew', 0.0)), y_offset=float(options_dict.get('adjust_ns', 0.0)), \
                                                         options_dict=options_dict, spline_configuration_json=spline_json, printf=printf, course_version=course_version)

        if len(osm_trees) > 0:
            printf("Adding trees from OpenStreetMap")
            for o in get_trees(course_json['theme'], options_dict.get('tree_variety', False), osm_trees, course_version):
                course_json[obj_tag].append(o)

    # Automatically adjust course elevation
    printf("Moving course to lowest valid elevation")
    course_json = tgc_tools.elevate_terrain(course_json, None, printf=printf, course_version=course_version)

    printf("Course Description Complete")

    return course_json

def generate_flat_course(course_json, xml_data, options_dict={}, printf=print, course_version=-1):
    course_json, osm_trees = OSMTGC.addOSMFromXML(course_json, xml_data, options_dict=options_dict, printf=printf,
        course_version=course_version)
    if course_version not in tgc_definitions.version_tags:
        print("invalid version")
        print(course_version)
        return

    obj_tag = tgc_definitions.version_tags[course_version]['objects']

    if len(osm_trees) > 0:
        printf("Adding trees from OpenStreetMap")
        for o in get_trees(course_json['theme'], options_dict.get('tree_variety', False), osm_trees, course_version):
            course_json[obj_tag].append(o)

    return course_json

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python program.py COURSE_DIRECTORY HEIGHTMAP_DIRECTORY")
        sys.exit(0)
    else:
        course_dir_path = sys.argv[1]
        heightmap_dir_path = sys.argv[2]

    print("Getting course description")
    course_json = tgc_tools.get_course_json(course_dir_path)

    print("Generating course")
    course_json = generate_course(course_json, heightmap_dir_path)

    print("Saving new course description")
    tgc_tools.write_course_json(course_dir_path, course_json)
