#!/usr/bin/env python3
"""
Localizes query images against a saved sparse 3D model and visualizes results.

This script:
1. Reuses existing SuperPoint features from the reference reconstruction
2. Extracts features for new query images
3. Matches query features against reference features
4. Localizes each query image using PnP+RANSAC
5. Visualizes camera poses and inlier points in 3D
6. Generates 2D correspondence visualizations

Usage:
    python localize_queries_v1.py
"""

from pathlib import Path
import pycolmap
import numpy as np
import shutil

from hloc import extract_features, match_features, pairs_from_exhaustive, visualization
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils import viz_3d
from hloc.utils.viz import plot_images, plot_matches, add_text, cm_RdGn
from hloc.utils.io import read_image
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation


def main():
    # ========== Setup Paths and Configuration ==========
    print("=" * 60)
    print("Query Image Localization Script")
    print("=" * 60)

    # Input paths
    reference_images = Path("final_inputs/v1/images")
    query_images = Path("test_images/images_v1")
    sparse_model = Path("final_inputs/v1/sparse/0")
    existing_features = Path("outputs/sfm_1s/feats-superpoint-n4096-r1024.h5")

    # Validate input paths
    if not reference_images.exists():
        raise FileNotFoundError(f"Reference images not found: {reference_images}")
    if not query_images.exists():
        raise FileNotFoundError(f"Query images not found: {query_images}")
    if not sparse_model.exists():
        raise FileNotFoundError(f"Sparse model not found: {sparse_model}")
    if not existing_features.exists():
        raise FileNotFoundError(f"Existing features not found: {existing_features}")

    # Output directory
    outputs = Path("outputs/localization_v1")
    outputs.mkdir(exist_ok=True, parents=True)

    # Output files
    features_path = outputs / "feats-superpoint-n4096-r1024.h5"
    pairs_path = outputs / "pairs-query-ref.txt"
    matches_path = outputs / "matches-superglue.h5"
    viz_2d_dir = outputs / "visualizations_2d"
    viz_2d_dir.mkdir(exist_ok=True)

    # Feature/matcher configs (matching pipeline_SfM.ipynb)
    feature_conf = extract_features.confs["superpoint_aachen"]  # n4096, r1024
    matcher_conf = match_features.confs["superglue"]            # outdoor weights

    print(f"\nInput paths:")
    print(f"  Reference images: {reference_images}")
    print(f"  Query images: {query_images}")
    print(f"  Sparse model: {sparse_model}")
    print(f"  Existing features: {existing_features}")
    print(f"\nOutput directory: {outputs}")

    # ========== Copy Existing Reference Features ==========
    print("\n" + "=" * 60)
    print("Step 1: Copying existing reference features")
    print("=" * 60)
    shutil.copy(existing_features, features_path)
    print(f"Copied features to {features_path}")

    # ========== Extract Features for Query Images ==========
    print("\n" + "=" * 60)
    print("Step 2: Extracting features for query images")
    print("=" * 60)

    # Get query image list
    query_list = sorted([p.name for p in query_images.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    print(f"Found {len(query_list)} query images: {query_list}")

    if len(query_list) == 0:
        raise ValueError(f"No query images found in {query_images}")

    # Extract query features (append to existing file)
    print(f"\nExtracting SuperPoint features for query images...")
    extract_features.main(
        feature_conf,
        query_images,
        image_list=query_list,
        feature_path=features_path,
        overwrite=False  # Append mode
    )
    print("Feature extraction complete")

    # ========== Generate Query-Reference Pairs ==========
    print("\n" + "=" * 60)
    print("Step 3: Generating query-reference pairs")
    print("=" * 60)

    # Load sparse model
    print(f"Loading sparse model from {sparse_model}...")
    model = pycolmap.Reconstruction(sparse_model)

    # Get reference image names from model
    ref_image_list = sorted([img.name for img in model.images.values()])
    print(f"Loaded sparse model:")
    print(f"  Images: {len(model.images)}")
    print(f"  3D points: {len(model.points3D)}")
    print(f"  Cameras: {len(model.cameras)}")

    # Create exhaustive pairs (each query vs all references)
    num_pairs = len(query_list) * len(ref_image_list)
    print(f"\nGenerating exhaustive pairs:")
    print(f"  {len(query_list)} queries × {len(ref_image_list)} references = {num_pairs} pairs")

    pairs_from_exhaustive.main(
        pairs_path,
        image_list=query_list,
        ref_list=ref_image_list
    )
    print(f"Pairs saved to {pairs_path}")

    # ========== Match Features with SuperGlue ==========
    print("\n" + "=" * 60)
    print("Step 4: Matching features with SuperGlue")
    print("=" * 60)
    print("This may take a few minutes...")

    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path
    )
    print(f"Matches saved to {matches_path}")

    # ========== Localize Query Images ==========
    print("\n" + "=" * 60)
    print("Step 5: Localizing query images")
    print("=" * 60)

    # Localization config (from demo.ipynb)
    loc_config = {
        "estimation": {
            "ransac": {
                "max_error": 12  # RANSAC inlier threshold in pixels
            }
        },
        "refinement": {
            "refine_focal_length": True,
            "refine_extra_params": True
        }
    }

    # Create localizer
    print("Creating localizer with PnP+RANSAC...")
    localizer = QueryLocalizer(model, loc_config)

    # Get reference image IDs
    ref_ids = [model.find_image_with_name(name).image_id for name in ref_image_list]

    # Localize each query
    results = {}
    logs = {}

    print("\nLocalizing query images:")
    for query_name in query_list:
        # Infer camera from EXIF
        camera = pycolmap.infer_camera_from_image(query_images / query_name)

        # Localize via PnP+RANSAC
        ret, log = pose_from_cluster(
            localizer,
            query_name,
            camera,
            ref_ids,
            features_path,
            matches_path
        )

        if ret is not None:
            results[query_name] = ret
            logs[query_name] = log
            print(f"  {query_name}: OK - {ret['num_inliers']}/{len(ret['inlier_mask'])} inliers")
        else:
            print(f"  {query_name}: FAILED - Localization failed")

    print(f"\nSuccessfully localized {len(results)}/{len(query_list)} images")

    if len(results) == 0:
        print("\nNo images were successfully localized. Exiting.")
        return

    # ========== Visualize 3D Results ==========
    print("\n" + "=" * 60)
    print("Step 6: Generating 3D visualization")
    print("=" * 60)

    # Initialize figure
    fig = viz_3d.init_figure()

    # Plot sparse reconstruction (reference points in red)
    print("Plotting sparse reconstruction...")
    viz_3d.plot_reconstruction(
        fig,
        model,
        color='rgba(255,0,0,0.5)',
        name="sparse_model",
        points_rgb=True
    )

    # Plot each localized query camera
    print("Plotting localized cameras and inlier points...")
    for query_name, ret in results.items():
        log = logs[query_name]

        # Calculate and print pose
        c2w = ret["cam_from_world"].inverse()
        pos = c2w.translation
        rot = Rotation.from_matrix(c2w.rotation.matrix())
        yaw, pitch, roll = rot.as_euler('zyx', degrees=True)
        
        print(f"\n  {query_name} Pose:")
        print(f"    Position (x, y, z): {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}")
        print(f"    Rotation (yaw, pitch, roll): {yaw:.4f}, {pitch:.4f}, {roll:.4f}")

        # Plot camera frustum (green)
        viz_3d.plot_camera_colmap(
            fig,
            ret["cam_from_world"],
            ret["camera"],
            color="rgba(0,255,0,0.5)",
            name=query_name,
            fill=True,
            text=f"{query_name}: {ret['num_inliers']}/{len(ret['inlier_mask'])} inliers"
        )

        # Plot inlier 3D points (lime)
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
    viz_3d_path = outputs / "visualization_3d.html"
    fig.write_html(str(viz_3d_path))
    print(f"\n[OK] Saved 3D visualization to {viz_3d_path}")

    # Try to display if running interactively
    try:
        fig.show()
    except:
        print("  (Open the HTML file in a browser to view the interactive 3D plot)")

    # ========== Visualize 2D Correspondences ==========
    print("\n" + "=" * 60)
    print("Step 7: Generating 2D correspondence visualizations")
    print("=" * 60)

    for query_name, log in logs.items():
        try:
            # Load query image
            q_image = read_image(query_images / query_name)

            # Get inlier mask and keypoints
            inliers = np.array(log["PnP_ret"]["inlier_mask"])
            mkp_q = log["keypoints_query"]
            n = len(log["db"])

            # Find database images and count inliers
            kp_idxs, kp_to_3D_to_db = log["keypoint_index_to_db"]
            counts = np.zeros(n)
            dbs_kp_q_db = [[] for _ in range(n)]
            inliers_dbs = [[] for _ in range(n)]

            for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers, kp_to_3D_to_db)):
                track = model.points3D[p3D_id].track
                track = {el.image_id: el.point2D_idx for el in track.elements}
                for db_idx in db_idxs:
                    counts[db_idx] += inl
                    kp_db = track[log["db"][db_idx]]
                    dbs_kp_q_db[db_idx].append((i, kp_db))
                    inliers_dbs[db_idx].append(inl)

            # Display top 2 database images with most inliers
            db_sort = np.argsort(-counts)
            for rank, db_idx in enumerate(db_sort[:2]):
                db = model.images[log["db"][db_idx]]
                db_name = db.name
                db_kp_q_db = np.array(dbs_kp_q_db[db_idx])

                if len(db_kp_q_db) == 0:
                    continue

                kp_q = mkp_q[db_kp_q_db[:, 0]]
                kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
                inliers_db = inliers_dbs[db_idx]

                # Load database image
                db_image = read_image(reference_images / db_name)

                # Create visualization
                color = cm_RdGn(inliers_db).tolist()
                text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

                plot_images([q_image, db_image], dpi=75)
                plot_matches(kp_q, kp_db, color, a=0.1)
                add_text(0, text)
                opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
                add_text(0, query_name, **opts)
                add_text(1, db_name, **opts)

                # Save figure
                output_name = f"{query_name[:-4]}_vs_{db_name[:-4]}_rank{rank}.jpg"
                output_path = viz_2d_dir / output_name
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                plt.close()

            print(f"  {query_name}: OK - Saved 2 visualizations to {viz_2d_dir}")
        except Exception as e:
            print(f"  {query_name}: FAILED - {e}")

    # ========== Save Results Summary ==========
    print("\n" + "=" * 60)
    print("Step 8: Saving results summary")
    print("=" * 60)

    summary_path = outputs / "localization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Query Image Localization Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sparse model: {sparse_model}\n")
        f.write(f"Reference images: {len(ref_image_list)}\n")
        f.write(f"Query images: {len(query_list)}\n")
        f.write(f"Successfully localized: {len(results)}/{len(query_list)}\n\n")

        for query_name in query_list:
            if query_name in results:
                ret = results[query_name]
                f.write(f"{query_name}:\n")
                f.write(f"  Inliers: {ret['num_inliers']}/{len(ret['inlier_mask'])}\n")
                f.write(f"  Camera: {ret['camera']}\n")
                f.write(f"  Pose (qvec): {ret['cam_from_world'].rotation.quat}\n")
                f.write(f"  Pose (tvec): {ret['cam_from_world'].translation}\n\n")
            else:
                f.write(f"{query_name}: FAILED\n\n")

    print(f"[OK] Saved summary to {summary_path}")

    # ========== Done ==========
    print("\n" + "=" * 60)
    print("Localization Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  3D visualization: {viz_3d_path}")
    print(f"  2D visualizations: {viz_2d_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Features: {features_path}")
    print(f"  Matches: {matches_path}")
    print("\nOpen visualization_3d.html in a browser to view the interactive 3D plot!")


if __name__ == "__main__":
    main()
