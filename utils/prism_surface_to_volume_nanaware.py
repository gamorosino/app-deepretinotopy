#!/usr/bin/env python3
"""Project native surface metrics into a FreeSurfer volume with neuropythy-style NaN handling.

Voxel addressing reuses the prism model in prism_surface_to_volume.py, which matches
neuropythy's Cortex.image_address. Interpolation follows neuropythy's
address_interpolate (method='linear'): a non-finite vertex value gets zero weight and the
remaining barycentric weights are renormalized, so a voxel is null only when all three
vertices of its triangle are non-finite. With method='nearest', the value of the vertex
with the largest barycentric weight is used as-is.

Example:
    python run_from_freesurfer/prism_surface_to_volume_nanaware.py \
        --freesurfer-subject-dir <fs_subject>/output \
        --lh-metric <native_metrics>/sub-X.polarAngle.lh.native.func.gii \
        --rh-metric <native_metrics>/sub-X.polarAngle.rh.native.func.gii \
        --output-volume polarAngle.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prism_surface_to_volume as psv  # noqa: E402


def build_hemisphere_address(
    freesurfer_subject_dir: Path,
    template_image: nib.spatialimages.SpatialImage,
    hemisphere: str,
    native_surface_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return prism voxel addresses of one hemisphere in the template image grid."""
    white_coordinates, pial_coordinates, faces, coordinates_are_scanner_ras = psv.load_surface_pair(
        freesurfer_subject_dir=freesurfer_subject_dir,
        hemisphere=hemisphere,
        native_surface_dir=native_surface_dir,
    )
    surface_to_voxel = psv.build_surface_to_voxel_affine(
        brain_image=template_image,
        freesurfer_subject_dir=freesurfer_subject_dir,
        coordinates_are_scanner_ras=coordinates_are_scanner_ras,
    )
    return psv.build_image_address(
        white_coordinates.T.astype(np.float64),
        pial_coordinates.T.astype(np.float64),
        faces.T.astype(np.int32),
        template_image.shape[:3],
        surface_to_voxel,
    )


def interpolate_face_values_nan_aware(
    address: dict[str, np.ndarray],
    surface_data: np.ndarray,
    method: str = "linear",
    null: float = np.nan,
) -> np.ndarray:
    """Interpolate per-vertex data at addressed voxels, ignoring non-finite vertices."""
    if method == "nearest":
        return psv.interpolate_face_values(address, surface_data, "nearest")
    if method != "linear":
        raise ValueError(f"Unsupported interpolation method: {method}")

    coordinates = address["coordinates"]
    if coordinates.size == 0:
        return np.zeros((0,), dtype=np.float64)

    weights = np.vstack([coordinates[0], coordinates[1], 1.0 - coordinates[0] - coordinates[1]])
    triangle_values = np.asarray(surface_data, dtype=np.float64)[address["faces"]]

    non_finite = ~np.isfinite(triangle_values)
    weights[non_finite] = 0.0
    triangle_values = np.where(non_finite, 0.0, triangle_values)

    weight_sum = np.sum(weights, axis=0)
    all_missing = np.isclose(weight_sum, 0.0)
    safe_sum = np.where(all_missing, 1.0, weight_sum)
    values = np.sum(triangle_values * weights, axis=0) / safe_sum
    values[all_missing] = null
    return values


def project_surface_metrics_to_volume(
    freesurfer_subject_dir: Path,
    hemisphere_metric_paths: dict[str, Path],
    output_path: Path,
    method: str = "linear",
    template_path: Path | None = None,
    native_surface_dir: Path | None = None,
    fill: float = 0.0,
    null: float = np.nan,
) -> Path:
    """Project lh/rh surface metrics into one volume aligned to the template image."""
    if not hemisphere_metric_paths:
        raise ValueError("At least one hemisphere metric is required")
    template_path = template_path or freesurfer_subject_dir / "mri" / "brain.mgz"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template image: {template_path}")

    template_image = nib.load(str(template_path))
    output_volume = np.full(template_image.shape[:3], fill, dtype=np.float32)
    for hemisphere, metric_path in hemisphere_metric_paths.items():
        address = build_hemisphere_address(freesurfer_subject_dir, template_image, hemisphere, native_surface_dir)
        surface_data = psv.load_morph_or_gifti(metric_path)
        values = interpolate_face_values_nan_aware(address, surface_data, method=method, null=null)
        output_volume[tuple(address["voxel_indices"])] = values.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(psv.make_output_image(output_volume, template_image), str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freesurfer-subject-dir", required=True, help="FreeSurfer subject directory containing mri/ and surf/.")
    parser.add_argument("--lh-metric", help="Left-hemisphere native surface metric (GIFTI or FreeSurfer morph).")
    parser.add_argument("--rh-metric", help="Right-hemisphere native surface metric (GIFTI or FreeSurfer morph).")
    parser.add_argument("--output-volume", required=True, help="Output .nii.gz path.")
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear", help="Interpolation method (default: linear).")
    parser.add_argument("--template", help="Template volume defining the output grid (default: <subject>/mri/brain.mgz).")
    parser.add_argument("--native-surface-dir", help="Optional directory with scanner-RAS native GIFTI white/pial surfaces.")
    parser.add_argument("--fill", type=float, default=0.0, help="Value for voxels outside the cortical ribbon (default: 0).")
    parser.add_argument("--null", type=float, default=np.nan, help="Value for ribbon voxels whose three vertices are all non-finite (default: nan).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hemisphere_metric_paths = {
        hemisphere: Path(path)
        for hemisphere, path in (("lh", args.lh_metric), ("rh", args.rh_metric))
        if path
    }
    if not hemisphere_metric_paths:
        raise ValueError("Provide --lh-metric and/or --rh-metric")

    output_path = project_surface_metrics_to_volume(
        freesurfer_subject_dir=Path(args.freesurfer_subject_dir),
        hemisphere_metric_paths=hemisphere_metric_paths,
        output_path=Path(args.output_volume),
        method=args.method,
        template_path=Path(args.template) if args.template else None,
        native_surface_dir=Path(args.native_surface_dir) if args.native_surface_dir else None,
        fill=args.fill,
        null=args.null,
    )
    print(output_path)


if __name__ == "__main__":
    main()
