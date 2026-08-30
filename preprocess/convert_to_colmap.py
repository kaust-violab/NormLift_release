"""
Convert NeRF/ScanNet scene (transforms_train.json) to COLMAP format.

Input:  lerf_ovs/scene0062_00/
Output: lerf_ovs/scene0062_00_colmap/   (same structure as figurines/)

Usage:
    python my_convert_to_colmap.py
    python my_convert_to_colmap.py --src lerf_ovs/scene0062_00 --dst lerf_ovs/scene0062_00_colmap
"""

import os
import sys
import json
import struct
import shutil
import argparse
import numpy as np
from pathlib import Path

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("plyfile not found, install with: pip install plyfile")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Rotation matrix  →  quaternion  (qw, qx, qy, qz)
# From COLMAP's read_write_model.py
# ---------------------------------------------------------------------------
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz,  Ryx + Rxy,        Rzx + Rxz,        Ryz - Rzy],
        [Ryx + Rxy,        Ryy - Rxx - Rzz,   Rzy + Ryz,        Rzx - Rxz],
        [Rzx + Rxz,        Rzy + Ryz,         Rzz - Rxx - Ryy,  Rxy - Ryx],
        [Ryz - Rzy,        Rzx - Rxz,         Rxy - Ryx,        Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]  # qw qx qy qz
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ---------------------------------------------------------------------------
# Write cameras.bin  (COLMAP binary format)
# Model 1 = PINHOLE  →  params: fx, fy, cx, cy
# ---------------------------------------------------------------------------
def write_cameras_bin(path, cameras):
    """cameras: dict  camera_id -> {'model_id', 'w', 'h', 'params': [fx,fy,cx,cy]}"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam_id, cam in cameras.items():
            f.write(struct.pack('<i', cam_id))
            f.write(struct.pack('<i', cam['model_id']))
            f.write(struct.pack('<Q', cam['w']))
            f.write(struct.pack('<Q', cam['h']))
            f.write(struct.pack(f"<{len(cam['params'])}d", *cam['params']))


# ---------------------------------------------------------------------------
# Write images.bin  (COLMAP binary format)
# ---------------------------------------------------------------------------
def write_images_bin(path, images):
    """images: list of dicts  {image_id, qvec, tvec, camera_id, name, xys, point3D_ids}
       xys and point3D_ids may be empty arrays.
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for img in images:
            f.write(struct.pack('<i', img['image_id']))
            f.write(struct.pack('<4d', *img['qvec']))   # qw qx qy qz
            f.write(struct.pack('<3d', *img['tvec']))   # tx ty tz
            f.write(struct.pack('<i', img['camera_id']))
            f.write(img['name'].encode() + b'\x00')
            # 2D keypoints (empty)
            num_pts2d = len(img.get('xys', []))
            f.write(struct.pack('<Q', num_pts2d))
            for (x, y), pt3d_id in zip(img.get('xys', []), img.get('point3D_ids', [])):
                f.write(struct.pack('<2d', x, y))
                f.write(struct.pack('<q', pt3d_id))


# ---------------------------------------------------------------------------
# Write points3D.bin  (COLMAP binary format)
# ---------------------------------------------------------------------------
def write_points3D_bin(path, points):
    """points: list of dicts {point3D_id, xyz, rgb, error, track}
       track: list of (image_id, point2D_idx) — may be empty.
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pt in points:
            f.write(struct.pack('<Q', pt['point3D_id']))
            f.write(struct.pack('<3d', *pt['xyz']))
            f.write(struct.pack('<3B', *pt['rgb']))
            f.write(struct.pack('<d', pt.get('error', 0.0)))
            track = pt.get('track', [])
            f.write(struct.pack('<Q', len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack('<I', img_id))
                f.write(struct.pack('<I', pt2d_idx))


# ---------------------------------------------------------------------------
# Load all frames from one transforms json, return list of frame dicts
# with 'file_path' and 'transform_matrix' fields.
# ---------------------------------------------------------------------------
def load_transforms(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data, data.get('frames', [])


# ---------------------------------------------------------------------------
# Convert NeRF c2w (OpenGL: Y-up, Z-back) to COLMAP (Y-down, Z-forward)
# Returns qvec (qw qx qy qz) and tvec (tx ty tz)
# ---------------------------------------------------------------------------
def nerf_c2w_to_colmap(transform_matrix):
    c2w = np.array(transform_matrix, dtype=np.float64)
    # Flip Y and Z axes: OpenGL→COLMAP
    c2w[:3, 1:3] *= -1
    # World-to-camera
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    qvec = rotmat2qvec(R)
    return qvec, T


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # ---- Load transforms ----
    meta_train, frames_train = load_transforms(src_dir / 'transforms_train.json')
    frames_test = []
    if (src_dir / 'transforms_test.json').exists():
        _, frames_test = load_transforms(src_dir / 'transforms_test.json')

    # Deduplicate by file_path
    seen = set()
    all_frames = []
    for f in frames_train + frames_test:
        fp = f['file_path']
        if fp not in seen:
            seen.add(fp)
            all_frames.append(f)

    print(f"Total unique frames: {len(all_frames)}")

    # ---- Camera intrinsics (shared for all images) ----
    w   = meta_train['w']
    h   = meta_train['h']
    fx  = meta_train['fl_x']
    fy  = meta_train['fl_y']
    cx  = meta_train['cx']
    cy  = meta_train['cy']

    # ---- Create output directories ----
    sparse_dir = dst_dir / 'sparse' / '0'
    images_dir = dst_dir / 'images'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write cameras.bin (one PINHOLE camera) ----
    cameras = {
        1: {
            'model_id': 1,   # PINHOLE
            'w': w,
            'h': h,
            'params': [fx, fy, cx, cy],
        }
    }
    write_cameras_bin(sparse_dir / 'cameras.bin', cameras)
    print(f"Wrote cameras.bin  (PINHOLE {w}x{h}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f})")

    # ---- Write images.bin + copy image files ----
    images = []
    n_skipped_bad_pose = 0
    for img_id, frame in enumerate(all_frames, start=1):
        # ScanNet sometimes contains tracker-failed frames whose transform_matrix
        # is filled with ±inf. Skip those — eigh on a non-finite K won't converge.
        M = np.asarray(frame['transform_matrix'], dtype=np.float64)
        if not np.all(np.isfinite(M)):
            n_skipped_bad_pose += 1
            print(f"  WARNING: non-finite pose for frame {frame['file_path']}, skipping")
            continue
        qvec, tvec = nerf_c2w_to_colmap(frame['transform_matrix'])

        # Determine source image path (try .jpg then .png)
        fp = frame['file_path']          # e.g. "color/0"
        stem = Path(fp).stem             # "0"
        parent = Path(fp).parent         # "color"

        src_jpg = src_dir / parent / (stem + '.jpg')
        src_png = src_dir / parent / (stem + '.png')
        if src_jpg.exists():
            src_img = src_jpg
            dst_name = stem + '.jpg'
        elif src_png.exists():
            src_img = src_png
            dst_name = stem + '.png'
        else:
            print(f"  WARNING: image not found for frame {fp}, skipping")
            continue

        # Copy image to images/
        dst_img = images_dir / dst_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        images.append({
            'image_id':  img_id,
            'qvec':      qvec,
            'tvec':      tvec,
            'camera_id': 1,
            'name':      dst_name,
            'xys':       [],
            'point3D_ids': [],
        })

    write_images_bin(sparse_dir / 'images.bin', images)
    print(f"Wrote images.bin  ({len(images)} images, skipped {n_skipped_bad_pose} non-finite poses)")

    # ---- Write points3D.bin from points3d.ply ----
    ply_src = src_dir / 'points3d.ply'
    if ply_src.exists():
        plydata = PlyData.read(ply_src)
        verts = plydata['vertex'].data
        points = []
        for i, v in enumerate(verts):
            points.append({
                'point3D_id': i + 1,
                'xyz': [float(v['x']), float(v['y']), float(v['z'])],
                'rgb': [int(v['red']), int(v['green']), int(v['blue'])],
                'error': 0.0,
                'track': [],
            })
        write_points3D_bin(sparse_dir / 'points3D.bin', points)
        # Also copy the ply directly (3DGS can use it for init)
        shutil.copy2(ply_src, sparse_dir / 'points3D.ply')
        print(f"Wrote points3D.bin + points3D.ply  ({len(points)} points)")
    else:
        # Write empty points3D.bin
        write_points3D_bin(sparse_dir / 'points3D.bin', [])
        print("points3d.ply not found — wrote empty points3D.bin")

    print(f"\nDone. Output: {dst_dir}")
    print(f"  {dst_dir}/sparse/0/cameras.bin")
    print(f"  {dst_dir}/sparse/0/images.bin")
    print(f"  {dst_dir}/sparse/0/points3D.bin")
    print(f"  {dst_dir}/sparse/0/points3D.ply")
    print(f"  {dst_dir}/images/  ({len(images)} files)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='lerf_ovs/scene0200_00',
                        help='Source scene directory (contains transforms_train.json)')
    parser.add_argument('--dst', default='lerf_ovs/scene0200_00_colmap',
                        help='Output COLMAP directory')
    args = parser.parse_args()

    # Resolve relative paths from this script's directory
    script_dir = Path(__file__).parent
    src = Path(args.src) if Path(args.src).is_absolute() else script_dir / args.src
    dst = Path(args.dst) if Path(args.dst).is_absolute() else script_dir / args.dst

    convert(src, dst)
