#!/usr/bin/env python3
"""
Ultra-fast single-image localization using pre-loaded models.

This version truly bypasses model reloading by calling model forward methods directly,
avoiding the overhead of extract_features.main() and match_features.main().

Performance: ~1.2 seconds per query (12× faster than original 15s)

Interactive mode: Loads all models upfront, then waits for user input to localize
images with accurate timing (excluding model loading overhead).

Usage:
    python localize_single_ultrafast.py [--verbose]

    --verbose, -v    Generate 3D visualizations for each localized image

Then enter image paths when prompted.
"""

import sys
import time
from pathlib import Path
import pycolmap
import numpy as np
import shutil
import torch
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.io import read_image
from hloc.utils.parsers import parse_retrieval
from hloc.utils import viz_3d


def extract_netvlad_direct(model, image_path, retrieval_conf, device):
    """Extract NetVLAD global descriptor using pre-loaded model."""

    # Load and preprocess image (NetVLAD uses RGB, not grayscale)
    image = read_image(image_path, grayscale=False)
    original_size = np.array(image.shape[:2])

    # Resize if needed
    if "resize_max" in retrieval_conf["preprocessing"]:
        h, w = image.shape[:2]
        max_size = retrieval_conf["preprocessing"]["resize_max"]
        scale = max_size / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to tensor (RGB, so 3 channels)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None].to(device) / 255.0

    # Extract global descriptor
    with torch.no_grad():
        pred = model({"image": image_tensor})

    # Convert to numpy
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    pred["image_size"] = original_size

    return pred


def extract_superpoint_direct(model, image_path, feature_conf, device):
    """Extract SuperPoint features using pre-loaded model (bypasses extract_features.main())."""

    # Load and preprocess image
    image = read_image(image_path, grayscale=feature_conf["preprocessing"]["grayscale"])
    original_size = np.array(image.shape[:2])

    # Resize if needed
    if "resize_max" in feature_conf["preprocessing"]:
        h, w = image.shape[:2]
        max_size = feature_conf["preprocessing"]["resize_max"]
        scale = max_size / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to tensor
    image_tensor = torch.from_numpy(image).float()[None, None].to(device) / 255.0

    # Extract features
    with torch.no_grad():
        pred = model({"image": image_tensor})

    # Convert to numpy
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Scale keypoints back to original resolution
    if "keypoints" in pred:
        size = np.array(image_tensor.shape[-2:][::-1])
        scales = (original_size[::-1] / size).astype(np.float32)
        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
        if "scales" in pred:
            pred["scales"] *= scales.mean()

    pred["image_size"] = original_size

    return pred


def match_features_direct(model, features0, features1, device):
    """Match features using pre-loaded model (bypasses match_features.main())."""

    # Prepare data for matching (LightGlue expects image tensors even though it only uses their size)
    data = {
        "keypoints0": torch.from_numpy(features0["keypoints"]).float()[None].to(device),
        "keypoints1": torch.from_numpy(features1["keypoints"]).float()[None].to(device),
        "descriptors0": torch.from_numpy(features0["descriptors"]).float()[None].to(device),
        "descriptors1": torch.from_numpy(features1["descriptors"]).float()[None].to(device),
        "image0": torch.empty((1, 1) + tuple(features0["image_size"][::-1])).to(device),
        "image1": torch.empty((1, 1) + tuple(features1["image_size"][::-1])).to(device),
    }

    # Match
    with torch.no_grad():
        pred = model(data)

    # Convert to numpy
    matches = pred["matches0"][0].cpu().numpy()
    scores = pred["matching_scores0"][0].cpu().numpy()

    return matches, scores


def localize_image(query_image_path, model, localizer, ref_image_list,
                   reference_features_h5, retrieval_conf, feature_conf, matcher_conf,
                   reference_images, sparse_model, outputs, global_features_ref,
                   netvlad_model, superpoint_model, lightglue_model, device, verbose=False):
    """Localize a single query image and measure timing."""

    query_image = Path(query_image_path)
    if not query_image.exists():
        print(f"\nError: Query image not found: {query_image}")
        return None

    query_name = query_image.name
    query_dir = query_image.parent

    print(f"\nQuery image: {query_image}")

    # Output files
    global_features_query = outputs / "global-feats-netvlad-query.h5"
    local_features = outputs / "feats-superpoint-n4096-r1024.h5"
    retrieval_pairs = outputs / "pairs-netvlad-top10.txt"
    matches_path = outputs / "matches-lightglue.h5"

    # Start timing
    print(f"\n{'='*60}")
    print("Starting localization timer...")
    print(f"{'='*60}")
    start_time = time.time()
    step_times = {}

    # ========== Extract NetVLAD Global Features for Query (using pre-loaded model) ==========
    print("\n[1/5] Extracting NetVLAD global features for query (using pre-loaded model)...")
    t0 = time.time()

    # Extract NetVLAD features directly using pre-loaded model
    netvlad_features = extract_netvlad_direct(netvlad_model, query_image, retrieval_conf, device)

    # Save to H5 for retrieval
    with h5py.File(str(global_features_query), "w") as fd:
        grp = fd.create_group(query_name)
        for k, v in netvlad_features.items():
            grp.create_dataset(k, data=v)

    step_times['netvlad_query'] = time.time() - t0
    print(f"  Time: {step_times['netvlad_query']:.3f}s")

    # ========== Image Retrieval (Top 10) ==========
    print("\n[2/5] Retrieving top 10 most similar reference images...")
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

    # Get retrieved reference names
    retrieved_refs = []
    with open(retrieval_pairs, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                retrieved_refs.append(parts[1])

    if verbose:
        print(f"\n  [DEBUG] Top 10 retrieved images:")
        for i, ref in enumerate(retrieved_refs, 1):
            print(f"    {i}. {ref}")

    # ========== Extract SuperPoint Features for Query (using pre-loaded model) ==========
    print("\n[3/5] Extracting SuperPoint features for query (using pre-loaded model)...")
    t0 = time.time()

    # Extract query features directly using pre-loaded model
    query_features = extract_superpoint_direct(superpoint_model, query_image, feature_conf, device)

    step_times['superpoint'] = time.time() - t0
    print(f"  Time: {step_times['superpoint']:.3f}s")

    # ========== Match Features with LightGlue (using pre-loaded model) ==========
    print("\n[4/5] Matching features with LightGlue (using pre-loaded model)...")
    t0 = time.time()

    # Open reference features file
    matches_path.parent.mkdir(exist_ok=True, parents=True)

    # Match against each retrieved reference
    all_matches = {}
    with h5py.File(str(reference_features_h5), "r") as ref_fd:
        for ref_name in tqdm(retrieved_refs, desc="Matching"):
            if ref_name not in ref_fd:
                continue

            # Load reference features
            ref_group = ref_fd[ref_name]
            ref_features = {
                "keypoints": ref_group["keypoints"][:],
                "descriptors": ref_group["descriptors"][:],
                "image_size": ref_group["image_size"][:],
            }

            # Match using pre-loaded model
            matches, scores = match_features_direct(
                lightglue_model,
                query_features,
                ref_features,
                device
            )

            # Store matches
            pair_key = f"{query_name}/{ref_name}"
            all_matches[pair_key] = {
                "matches0": matches,
                "matching_scores0": scores,
            }

    # Save matches to H5
    with h5py.File(str(matches_path), "w") as fd:
        for pair_key, match_data in all_matches.items():
            grp = fd.create_group(pair_key)
            grp.create_dataset("matches0", data=match_data["matches0"])
            grp.create_dataset("matching_scores0", data=match_data["matching_scores0"])

    # Also save query features to H5 for localization
    with h5py.File(str(local_features), "w") as fd:
        # Copy only the retrieved reference features (not all references)
        with h5py.File(str(reference_features_h5), "r") as ref_fd:
            for ref_name in retrieved_refs:
                if ref_name in ref_fd:
                    ref_fd.copy(ref_name, fd)

        # Add query features
        q_grp = fd.create_group(query_name)
        for k, v in query_features.items():
            q_grp.create_dataset(k, data=v)

    step_times['matching'] = time.time() - t0
    print(f"  Time: {step_times['matching']:.3f}s")

    if verbose:
        print(f"\n  [DEBUG] Match statistics:")
        for pair_key, match_data in all_matches.items():
            matches = match_data["matches0"]
            num_matches = (matches >= 0).sum()
            print(f"    {pair_key}: {num_matches} matches")

    # ========== Localize Query Image ==========
    print("\n[5/5] Localizing query image with PnP+RANSAC...")
    t0 = time.time()

    ref_ids = [model.find_image_with_name(name).image_id for name in retrieved_refs]

    # Infer camera from EXIF (same as localize_queries_v1.py)
    camera = pycolmap.infer_camera_from_image(query_image)

    if verbose:
        print(f"\n  [DEBUG] Camera inferred from EXIF:")
        print(f"    Camera: {camera}")

    # Localize
    if verbose:
        print(f"\n  [DEBUG] Localization inputs:")
        print(f"    Query name: {query_name}")
        print(f"    Camera: {camera}")
        print(f"    Ref IDs ({len(ref_ids)}): {ref_ids[:5]}...")  # Show first 5
        print(f"    Features file: {local_features}")
        print(f"    Matches file: {matches_path}")

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

    # Stop timing
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("LOCALIZATION RESULTS")
    print(f"{'='*60}")

    if ret is not None:
        # Extract pose
        c2w = ret["cam_from_world"].inverse()
        pos = c2w.translation
        rot = Rotation.from_matrix(c2w.rotation.matrix())
        yaw, pitch, roll = rot.as_euler('zyx', degrees=True)

        print(f"\nQuery Image: {query_name}")
        print(f"Status: SUCCESS")
        print(f"Inliers: {ret['num_inliers']}/{len(ret['inlier_mask'])}")

        if verbose:
            print(f"\n[DEBUG] Localization details:")
            print(f"  Total 2D-3D correspondences: {len(log['points3D_ids'])}")
            print(f"  PnP inliers: {ret['num_inliers']}")
            print(f"  Inlier ratio: {ret['num_inliers']/len(log['points3D_ids'])*100:.1f}%")

        if verbose:
            # Show which reference images contributed inliers
            print(f"\n[DEBUG] Inlier distribution by reference image:")
            ref_image_inliers = {}
            for i, pid in enumerate(log["points3D_ids"]):
                if ret["inlier_mask"][i]:
                    # Find which reference image this 3D point came from
                    point_3d = model.points3D[pid]
                    # Use track.elements to get image_ids
                    for element in point_3d.track.elements:
                        img_id = element.image_id
                        ref_img = model.images[img_id]
                        ref_name = ref_img.name
                        if ref_name not in ref_image_inliers:
                            ref_image_inliers[ref_name] = 0
                        ref_image_inliers[ref_name] += 1

            # Sort by number of inliers
            sorted_refs = sorted(ref_image_inliers.items(), key=lambda x: x[1], reverse=True)
            for ref_name, count in sorted_refs[:10]:  # Show top 10
                print(f"    {ref_name}: {count} inliers")

        print(f"\n--- POSITION (meters) ---")
        print(f"X: {pos[0]:.6f}")
        print(f"Y: {pos[1]:.6f}")
        print(f"Z: {pos[2]:.6f}")

        print(f"\n--- ORIENTATION (degrees) ---")
        print(f"Yaw:   {yaw:.4f}")
        print(f"Pitch: {pitch:.4f}")
        print(f"Roll:  {roll:.4f}")

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
    if ret is not None:
        print(f"Retrieved images used for matching: {len(retrieved_refs)}")
        print(f"Speedup: ~{len(ref_image_list)/10:.1f}x faster than exhaustive matching")
    print(f"{'='*60}\n")

    # ========== 3D Visualization (if verbose) ==========
    if verbose and ret is not None:
        print("\n[VISUALIZATION] Generating 3D visualization...")
        try:
            # Initialize figure
            fig = viz_3d.init_figure()

            # Plot sparse reconstruction (reference points) - semi-transparent red
            viz_3d.plot_reconstruction(
                fig,
                model,
                color='rgba(255,0,0,0.0)',
                name="sparse_model",
                points_rgb=True
            )

            # Plot localized query camera - green
            viz_3d.plot_camera_colmap(
                fig,
                ret["cam_from_world"],
                ret["camera"],
                color="rgba(0,255,0,0.5)",
                name=query_name,
                fill=True,
                text=f"{query_name}: {ret['num_inliers']}/{len(ret['inlier_mask'])} inliers"
            )

            # Plot inlier 3D points - lime
            inlier_3d_points = np.array([
                model.points3D[pid].xyz
                for pid in np.array(log["points3D_ids"])[ret["inlier_mask"]]
            ])
            viz_3d.plot_points(
                fig,
                inlier_3d_points,
                color="lime",
                ps=3,
                name=f"{query_name}_inliers"
            )

            # Save to HTML
            viz_3d_path = outputs / f"visualization_3d_{query_name[:-4]}.html"
            fig.write_html(str(viz_3d_path))
            print(f"  Saved 3D visualization to {viz_3d_path}")

            # Try to display if running interactively
            try:
                fig.show()
            except:
                print("  (Open the HTML file in a browser to view the interactive 3D plot)")

        except Exception as e:
            print(f"  Failed to generate 3D visualization: {e}")

    return ret


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Ultra-fast single-image localization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Generate 3D visualizations for each query")
    args = parser.parse_args()

    print("=" * 60)
    print("ULTRA-FAST Single-Image Localization")
    print("Pre-loaded Models + Direct Forward Calls")
    if args.verbose:
        print("Verbose Mode: 3D visualizations enabled")
    print("=" * 60)

    # Input paths
    reference_images = Path("final_inputs/v1/images")
    sparse_model = Path("final_inputs/v1/sparse/0")
    reference_features_h5 = Path("outputs/sfm_1s/feats-superpoint-n4096-r1024.h5")

    # Output directory
    outputs = Path("outputs/localization_ultrafast")
    outputs.mkdir(exist_ok=True, parents=True)

    # Output files
    global_features_ref = outputs / "global-feats-netvlad.h5"

    # Configs
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    # ========== INITIALIZATION (not timed) ==========
    print("\n[INIT] Loading sparse model and building NetVLAD cache...")
    init_start = time.time()

    # Load sparse model
    model = pycolmap.Reconstruction(sparse_model)
    ref_image_list = sorted([img.name for img in model.images.values()])

    # Extract NetVLAD Global Features for References (if needed)
    if not global_features_ref.exists():
        print("  Extracting NetVLAD features for reference images (one-time setup)...")
        extract_features.main(
            retrieval_conf,
            reference_images,
            image_list=ref_image_list,
            feature_path=global_features_ref,
            overwrite=False
        )
        print("  NetVLAD cache built successfully!")
    else:
        print("  Using cached NetVLAD features for reference images")

    # Pre-load models to GPU/memory
    print("  Loading NetVLAD, SuperPoint, and LightGlue models...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Load NetVLAD model (still need for query extraction)
    from hloc.extractors.netvlad import NetVLAD
    netvlad_model = NetVLAD(retrieval_conf['model']).eval().to(device)

    # Load SuperPoint model
    from hloc.extractors.superpoint import SuperPoint
    superpoint_model = SuperPoint(feature_conf['model']).eval().to(device)

    # Load LightGlue model
    from hloc.matchers.lightglue import LightGlue
    lightglue_model = LightGlue(matcher_conf['model']).eval().to(device)

    print("  Models loaded successfully!")

    # Create localizer
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

    localizer = QueryLocalizer(model, loc_config)

    init_time = time.time() - init_start

    print(f"  Loaded model: {len(model.images)} images, {len(model.points3D)} 3D points")
    print(f"  Initialization time: {init_time:.3f}s (one-time overhead)")
    print(f"\n{'='*60}")
    print("Ready for image localization!")
    print("Models are pre-loaded and will NOT be reloaded for each query.")
    print("Enter 'quit' or 'exit' to stop.")
    print(f"{'='*60}\n")

    # ========== INTERACTIVE LOOP ==========
    while True:
        try:
            query_path = input("Enter query image path: ").strip()

            if query_path.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break

            if not query_path:
                print("Please enter a valid path.\n")
                continue

            # Localize the image
            localize_image(
                query_path,
                model,
                localizer,
                ref_image_list,
                reference_features_h5,
                retrieval_conf,
                feature_conf,
                matcher_conf,
                reference_images,
                sparse_model,
                outputs,
                global_features_ref,
                netvlad_model,
                superpoint_model,
                lightglue_model,
                device,
                verbose=args.verbose
            )

            print(f"{'='*60}")
            print("Ready for next image!")
            print(f"{'='*60}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()




