#!/usr/bin/env python3
"""
Fast single-image localization using HLOC with NetVLAD image retrieval.

This script localizes a single query image against a pre-built sparse 3D model
using image retrieval to speed up the process. No visualization - just prints
the camera pose and timing information.

Usage:
    python localize_single_fast.py <query_image_path>

Example:
    python localize_single_fast.py test_images/images_v1/1.jpg
"""

import sys
import time
from pathlib import Path
import pycolmap
import numpy as np

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster


def quaternion_to_euler(quat):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees.

    Args:
        quat: numpy array [w, x, y, z] (COLMAP format)

    Returns:
        (roll, pitch, yaw) in degrees
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def main(query_image_path):
    # ========== Configuration ==========
    print("=" * 60)
    print("Fast Single-Image Localization with NetVLAD Retrieval")
    print("=" * 60)

    # Convert to Path object
    query_image = Path(query_image_path)
    if not query_image.exists():
        print(f"\nError: Query image not found: {query_image}")
        return

    query_name = query_image.name
    query_dir = query_image.parent

    print(f"\nQuery image: {query_image}")

    # Input paths
    reference_images = Path("final_inputs/v1/images")
    sparse_model = Path("final_inputs/v1/sparse/0")
    existing_features = Path("outputs/sfm_1s/feats-superpoint-n4096-r1024.h5")

    # Output directory
    outputs = Path("outputs/localization_fast")
    outputs.mkdir(exist_ok=True, parents=True)

    # Output files
    global_features_ref = outputs / "global-feats-netvlad.h5"
    global_features_query = outputs / "global-feats-netvlad-query.h5"
    local_features = outputs / "feats-superpoint-n4096-r1024.h5"
    retrieval_pairs = outputs / "pairs-netvlad-top10.txt"
    matches_path = outputs / "matches-lightglue.h5"

    # Configs
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superpoint+lightglue"]  # Faster than SuperGlue

    # ========== Load Sparse Model ==========
    print("\n[1/7] Loading sparse model...")
    model = pycolmap.Reconstruction(sparse_model)
    ref_image_list = sorted([img.name for img in model.images.values()])
    print(f"  Loaded model: {len(model.images)} images, {len(model.points3D)} 3D points")

    # ========== Start Timing ==========
    print(f"\n{'='*60}")
    print("Starting localization timer...")
    print(f"{'='*60}")
    start_time = time.time()
    step_times = {}

    # ========== Extract NetVLAD Global Features for Query ==========
    print("\n[2/7] Extracting NetVLAD global features for query...")
    t0 = time.time()
    extract_features.main(
        retrieval_conf,
        query_dir,
        image_list=[query_name],
        feature_path=global_features_query,
        overwrite=True
    )
    step_times['netvlad_query'] = time.time() - t0
    print(f"  Time: {step_times['netvlad_query']:.3f}s")

    # ========== Extract NetVLAD Global Features for References (if needed) ==========
    if not global_features_ref.exists():
        print("\n[3/7] Extracting NetVLAD global features for reference images...")
        print("  (This is a one-time operation, will be cached for future runs)")
        t0 = time.time()
        extract_features.main(
            retrieval_conf,
            reference_images,
            image_list=ref_image_list,
            feature_path=global_features_ref,
            overwrite=False
        )
        step_times['netvlad_refs'] = time.time() - t0
        print(f"  Time: {step_times['netvlad_refs']:.3f}s")
    else:
        print("\n[3/7] Using cached NetVLAD features for reference images")

    # ========== Image Retrieval (Top 10) ==========
    print("\n[4/7] Retrieving top 10 most similar reference images...")
    t0 = time.time()
    pairs_from_retrieval.main(
        global_features_query,
        retrieval_pairs,
        num_matched=10,  # Top 10 matches
        db_descriptors=global_features_ref,
        db_model=sparse_model,
        query_prefix=None,
        query_list=[query_name]
    )
    step_times['retrieval'] = time.time() - t0
    print(f"  Time: {step_times['retrieval']:.3f}s")

    # Read pairs to see which references were selected
    with open(retrieval_pairs, 'r') as f:
        pairs = f.readlines()
    print(f"  Retrieved {len(pairs)} pairs (query vs top 10 references)")

    # ========== Extract SuperPoint Features for Query ==========
    print("\n[5/7] Extracting SuperPoint features for query...")
    t0 = time.time()

    # Copy existing reference features
    import shutil
    shutil.copy(existing_features, local_features)

    # Extract query features (append to existing)
    extract_features.main(
        feature_conf,
        query_dir,
        image_list=[query_name],
        feature_path=local_features,
        overwrite=False
    )
    step_times['superpoint'] = time.time() - t0
    print(f"  Time: {step_times['superpoint']:.3f}s")

    # ========== Match Features with LightGlue ==========
    print("\n[6/7] Matching features with LightGlue (faster than SuperGlue)...")
    t0 = time.time()
    match_features.main(
        matcher_conf,
        retrieval_pairs,
        features=local_features,
        matches=matches_path,
        overwrite=True  # Always rematch for new query
    )
    step_times['matching'] = time.time() - t0
    print(f"  Time: {step_times['matching']:.3f}s")

    # ========== Localize Query Image ==========
    print("\n[7/7] Localizing query image with PnP+RANSAC...")
    t0 = time.time()

    # Localization config
    loc_config = {
        "estimation": {
            "ransac": {
                "max_error": 12
            }
        },
        "refinement": {
            "refine_focal_length": True,
            "refine_extra_params": True
        }
    }

    # Create localizer
    localizer = QueryLocalizer(model, loc_config)

    # Get reference image IDs (only the top 10 retrieved)
    retrieved_refs = []
    with open(retrieval_pairs, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                retrieved_refs.append(parts[1])

    ref_ids = [model.find_image_with_name(name).image_id for name in retrieved_refs]

    # Infer camera from EXIF
    camera = pycolmap.infer_camera_from_image(query_image)

    # Localize
    ret, log = pose_from_cluster(
        localizer,
        query_name,
        camera,
        ref_ids,
        local_features,
        matches_path
    )
    step_times['pnp_ransac'] = time.time() - t0
    print(f"  Time: {step_times['pnp_ransac']:.3f}s")

    # ========== Stop Timing ==========
    end_time = time.time()
    elapsed_time = end_time - start_time

    # ========== Print Results ==========
    print(f"\n{'='*60}")
    print("LOCALIZATION RESULTS")
    print(f"{'='*60}")

    if ret is not None:
        # Extract pose
        cam_from_world = ret["cam_from_world"]
        position = cam_from_world.translation  # [x, y, z]
        rotation_quat = cam_from_world.rotation.quat  # [w, x, y, z]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(rotation_quat)

        print(f"\nQuery Image: {query_name}")
        print(f"Status: SUCCESS")
        print(f"Inliers: {ret['num_inliers']}/{len(ret['inlier_mask'])}")

        print(f"\n--- POSITION (meters) ---")
        print(f"X: {position[0]:.6f}")
        print(f"Y: {position[1]:.6f}")
        print(f"Z: {position[2]:.6f}")

        print(f"\n--- ORIENTATION (degrees) ---")
        print(f"Roll:  {roll:.2f}°")
        print(f"Pitch: {pitch:.2f}°")
        print(f"Yaw:   {yaw:.2f}°")

        print(f"\n--- ORIENTATION (quaternion) ---")
        print(f"w: {rotation_quat[0]:.6f}")
        print(f"x: {rotation_quat[1]:.6f}")
        print(f"y: {rotation_quat[2]:.6f}")
        print(f"z: {rotation_quat[3]:.6f}")

        print(f"\n--- CAMERA PARAMETERS ---")
        print(f"{ret['camera']}")

        print(f"\n--- TIMING ---")
        print(f"Total localization time: {elapsed_time:.3f} seconds")
        print(f"  ({elapsed_time*1000:.1f} ms)")

        # Detailed breakdown
        print(f"\nDetailed breakdown:")
        print(f"  - NetVLAD query extraction:  {step_times.get('netvlad_query', 0):.3f}s")
        print(f"  - Image retrieval:           {step_times.get('retrieval', 0):.3f}s")
        print(f"  - SuperPoint extraction:     {step_times.get('superpoint', 0):.3f}s")
        print(f"  - LightGlue matching:        {step_times.get('matching', 0):.3f}s")
        print(f"  - PnP+RANSAC:                {step_times.get('pnp_ransac', 0):.3f}s")
        if 'netvlad_refs' in step_times:
            print(f"  - NetVLAD refs (one-time):   {step_times['netvlad_refs']:.3f}s")

    else:
        print(f"\nQuery Image: {query_name}")
        print(f"Status: FAILED")
        print(f"Localization failed - could not estimate camera pose")
        print(f"\nPossible reasons:")
        print(f"  - Not enough feature matches with retrieved images")
        print(f"  - Query image too different from reference images")
        print(f"  - RANSAC could not find consistent pose")

        print(f"\n--- TIMING ---")
        print(f"Time until failure: {elapsed_time:.3f} seconds")

    print(f"\n{'='*60}")
    print(f"Retrieved images used for matching: {len(retrieved_refs)}")
    print(f"Speedup: ~{len(ref_image_list)/10:.1f}x faster than exhaustive matching")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python localize_single_fast.py <query_image_path>")
        print("\nExample:")
        print("  python localize_single_fast.py test_images/images_v1/1.jpg")
        sys.exit(1)

    query_path = sys.argv[1]
    main(query_path)
