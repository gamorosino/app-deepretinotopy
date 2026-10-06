#!/usr/bin/env python3
"""Project native pRF surface predictions into volume using a neuropythy-like prism model."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import io as freesurfer_io


METRIC_METHODS = {
    "polarAngle": "linear",
    "eccentricity": "linear",
    "pRFsize": "linear",
    "rfWidth": "linear",
    "r2": "nearest",
    "varea": "nearest",
}


def det_4x3(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, point_d: np.ndarray) -> np.ndarray:
    """Return the determinant of a 4x3 homogeneous point matrix."""
    return (
        point_a[1] * point_b[2] * point_c[0]
        + point_a[2] * point_b[0] * point_c[1]
        - point_a[2] * point_b[1] * point_c[0]
        - point_a[0] * point_b[2] * point_c[1]
        - point_a[1] * point_b[0] * point_c[2]
        + point_a[0] * point_b[1] * point_c[2]
        + point_a[2] * point_b[1] * point_d[0]
        - point_a[1] * point_b[2] * point_d[0]
        - point_a[2] * point_c[1] * point_d[0]
        + point_b[2] * point_c[1] * point_d[0]
        + point_a[1] * point_c[2] * point_d[0]
        - point_b[1] * point_c[2] * point_d[0]
        - point_a[2] * point_b[0] * point_d[1]
        + point_a[0] * point_b[2] * point_d[1]
        + point_a[2] * point_c[0] * point_d[1]
        - point_b[2] * point_c[0] * point_d[1]
        - point_a[0] * point_c[2] * point_d[1]
        + point_b[0] * point_c[2] * point_d[1]
        + point_a[1] * point_b[0] * point_d[2]
        - point_a[0] * point_b[1] * point_d[2]
        - point_a[1] * point_c[0] * point_d[2]
        + point_b[1] * point_c[0] * point_d[2]
        + point_a[0] * point_c[1] * point_d[2]
        - point_b[0] * point_c[1] * point_d[2]
    )


def tetrahedral_barycentric_coordinates(tetrahedron: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return tetrahedral barycentric coordinates for a vector of points."""
    tetrahedron = np.asarray(tetrahedron, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if tetrahedron.shape[0] != 4:
        raise ValueError(f"Expected tetrahedron with shape (4, 3, n) or (4, 3), got {tetrahedron.shape}")
    if tetrahedron.shape[1] != 3:
        tetrahedron = np.transpose(tetrahedron, (0, 2, 1))
    if points.shape[0] != 3:
        points = points.T

    determinant = det_4x3(tetrahedron[0], tetrahedron[1], tetrahedron[2], tetrahedron[3])
    det_0 = det_4x3(points, tetrahedron[1], tetrahedron[2], tetrahedron[3])
    det_1 = det_4x3(tetrahedron[0], points, tetrahedron[2], tetrahedron[3])
    det_2 = det_4x3(tetrahedron[0], tetrahedron[1], points, tetrahedron[3])
    det_3 = det_4x3(tetrahedron[0], tetrahedron[1], tetrahedron[2], points)

    determinant_sign = np.sign(determinant)
    invalid = np.logical_or(
        np.any([determinant_sign * sign == -1 for sign in np.sign([det_0, det_1, det_2, det_3])], axis=0),
        np.isclose(determinant, 0),
    )
    valid = np.logical_not(invalid)
    inverse_determinant = valid / (valid * determinant + invalid)
    return np.asarray([inverse_determinant * component for component in (det_0, det_1, det_2, det_3)])


PRISM_TETRAHEDRONS = np.array(
    [
        [0, 1, 6, 9],
        [1, 2, 7, 9],
        [2, 0, 8, 9],
        [0, 6, 8, 9],
        [1, 6, 7, 9],
        [2, 7, 8, 9],
        [0, 6, 8, 3],
        [1, 6, 7, 4],
        [2, 7, 8, 5],
        [6, 7, 8, 9],
    ],
    dtype=np.int32,
)
PRISM_WEIGHT_COUNTS = np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 3], dtype=np.int32)
PRISM_WEIGHTS = {
    1: np.reshape(np.arange(6, dtype=np.int32), (6, 1)),
    2: np.array(
        [
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [3, 4],
            [4, 5],
            [5, 3],
        ],
        dtype=np.int32,
    ),
    3: np.full((10, 3), -1, dtype=np.int32),
}
PRISM_WEIGHTS[3][-1] = [0, 1, 2]
PRISM_TETRAHEDRON_FLAT = PRISM_TETRAHEDRONS.flatten()
PRISM_TETRAHEDRON_WEIGHT_COUNT = PRISM_WEIGHT_COUNTS[PRISM_TETRAHEDRON_FLAT]
PRISM_WEIGHT_PLAN = tuple(
    (
        weight_count,
        indices,
        weight_table[PRISM_TETRAHEDRON_FLAT[indices]],
    )
    for weight_count, weight_table in PRISM_WEIGHTS.items()
    for indices in np.where(PRISM_TETRAHEDRON_WEIGHT_COUNT == weight_count)
)


def prism_barycentric_coordinates(triangle_one: np.ndarray, triangle_two: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return prism coordinates as (a, b, depth)."""
    triangle_one = np.asarray(triangle_one, dtype=np.float64)
    triangle_two = np.asarray(triangle_two, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    if triangle_one.shape[0] != 3:
        triangle_one = np.transpose(triangle_one, (1, 0) if triangle_one.ndim == 2 else (2, 0, 1))
    elif triangle_one.shape[1] != 3:
        triangle_one = np.transpose(triangle_one, (0, 2, 1))
    if triangle_two.shape[0] != 3:
        triangle_two = np.transpose(triangle_two, (1, 0) if triangle_two.ndim == 2 else (2, 0, 1))
    elif triangle_two.shape[1] != 3:
        triangle_two = np.transpose(triangle_two, (0, 2, 1))
    if points.shape[0] != 3:
        points = points.T

    (point_a1, point_b1, point_c1), (point_a2, point_b2, point_c2) = triangle_one, triangle_two
    midpoint_ab2, midpoint_bc2, midpoint_ca2 = np.mean([(point_a2, point_b2), (point_b2, point_c2), (point_c2, point_a2)], axis=1)
    midpoint_triangle_one = np.mean(triangle_one, axis=0)
    tetra_points = np.asarray(
        [
            point_a1,
            point_b1,
            point_c1,
            point_a2,
            point_b2,
            point_c2,
            midpoint_ab2,
            midpoint_bc2,
            midpoint_ca2,
            midpoint_triangle_one,
        ],
        dtype=np.float64,
    )
    barycentric_values = np.vstack(
        [tetrahedral_barycentric_coordinates(tetra_points[tetrahedron], points) for tetrahedron in PRISM_TETRAHEDRONS]
    )

    accumulated = np.zeros((6, barycentric_values.shape[1]), dtype=np.float64)
    for weight_count, indices, weight_indices in PRISM_WEIGHT_PLAN:
        for barycentric_index, output_index in zip(indices, weight_indices):
            accumulated[output_index] += barycentric_values[barycentric_index] / weight_count

    coordinates = accumulated[:3] + accumulated[3:]
    total = np.sum(coordinates, axis=0)
    coordinates[2] = 1.0 - np.sum(accumulated[:3], axis=0)
    invalid = np.where(~np.isclose(total, 1.0))[0]
    coordinates[:, invalid] = 0.0
    return coordinates


def load_morph_or_gifti(metric_path: Path) -> np.ndarray:
    """Load FreeSurfer morph data, falling back to GIFTI if needed."""
    if metric_path.exists():
        if metric_path.suffix == ".gii":
            image = nib.load(str(metric_path))
            if not image.darrays:
                raise ValueError(f"No data arrays found in {metric_path}")
            return np.asarray(image.darrays[0].data, dtype=np.float64).reshape(-1)
        return np.asarray(freesurfer_io.read_morph_data(str(metric_path)), dtype=np.float64)

    gifti_path = metric_path.with_suffix(metric_path.suffix + ".gii") if metric_path.suffix else Path(f"{metric_path}.gii")
    if not gifti_path.exists():
        raise FileNotFoundError(f"Missing surface metric: {metric_path} or {gifti_path}")

    image = nib.load(str(gifti_path))
    if not image.darrays:
        raise ValueError(f"No data arrays found in {gifti_path}")
    return np.asarray(image.darrays[0].data, dtype=np.float64).reshape(-1)


def apply_affine(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to 3xN coordinates."""
    homogeneous = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float64)])
    return (matrix @ homogeneous)[:3]


def make_translation_affine(offset: np.ndarray) -> np.ndarray:
    """Return a 4x4 affine that translates by an RAS offset."""
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = np.asarray(offset, dtype=np.float64).reshape(3)
    return affine


def load_c_ras(freesurfer_subject_dir: Path) -> np.ndarray:
    """Read the FreeSurfer c_ras offset used to align raw surfaces to scanner RAS."""
    candidate_paths = (
        freesurfer_subject_dir / "mri" / "brain.finalsurfs.mgz",
        freesurfer_subject_dir / "mri" / "T1.mgz",
        freesurfer_subject_dir / "mri" / "brain.mgz",
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            image = nib.load(str(candidate_path))
            return np.asarray(image.header["Pxyz_c"], dtype=np.float64).reshape(3)

    searched_paths = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Unable to derive c_ras. Searched:\n{searched_paths}")


def load_gifti_surface(surface_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinates and faces from a GIFTI surface."""
    image = nib.load(str(surface_path))
    if len(image.darrays) < 2:
        raise ValueError(f"Expected at least two GIFTI arrays in {surface_path}")
    coordinates = np.asarray(image.darrays[0].data, dtype=np.float64)
    faces = np.asarray(image.darrays[1].data, dtype=np.int32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape in {surface_path}: {coordinates.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face shape in {surface_path}: {faces.shape}")
    return coordinates, faces


def find_native_gifti_surface(native_surface_dir: Path | None, hemisphere: str, surface_name: str) -> Path | None:
    """Find a c_ras-aligned native GIFTI surface when one is available."""
    if native_surface_dir is None:
        return None
    hemisphere_letter = "L" if hemisphere == "lh" else "R"
    candidates = sorted(native_surface_dir.glob(f"*.{hemisphere_letter}.{surface_name}.native.surf.gii"))
    candidates.extend(sorted(native_surface_dir.glob(f"{hemisphere}.{surface_name}.native.surf.gii")))
    for candidate_path in candidates:
        if candidate_path.exists():
            return candidate_path
    return None


def load_surface_pair(
    freesurfer_subject_dir: Path,
    hemisphere: str,
    native_surface_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load white/pial surfaces and report whether coordinates are already scanner RAS."""
    native_white_path = find_native_gifti_surface(native_surface_dir, hemisphere, "white")
    native_pial_path = find_native_gifti_surface(native_surface_dir, hemisphere, "pial")
    if native_white_path is not None and native_pial_path is not None:
        white_coordinates, faces = load_gifti_surface(native_white_path)
        pial_coordinates, pial_faces = load_gifti_surface(native_pial_path)
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"White and pial GIFTI faces differ for hemisphere {hemisphere}")
        return white_coordinates, pial_coordinates, faces, True

    white_surface_path = freesurfer_subject_dir / "surf" / f"{hemisphere}.white"
    pial_surface_path = freesurfer_subject_dir / "surf" / f"{hemisphere}.pial"
    if not white_surface_path.exists():
        raise FileNotFoundError(f"Missing white surface: {white_surface_path}")
    if not pial_surface_path.exists():
        raise FileNotFoundError(f"Missing pial surface: {pial_surface_path}")

    white_coordinates, faces = freesurfer_io.read_geometry(str(white_surface_path))
    pial_coordinates, pial_faces = freesurfer_io.read_geometry(str(pial_surface_path))
    if not np.array_equal(faces, pial_faces):
        raise ValueError(f"White and pial faces differ for hemisphere {hemisphere}")
    return white_coordinates, pial_coordinates, faces, False


def build_surface_to_voxel_affine(
    brain_image: nib.spatialimages.SpatialImage,
    freesurfer_subject_dir: Path,
    coordinates_are_scanner_ras: bool,
) -> np.ndarray:
    """Build the surface-to-voxel affine with explicit c_ras handling."""
    if coordinates_are_scanner_ras:
        return np.linalg.inv(brain_image.affine)

    c_ras_affine = make_translation_affine(load_c_ras(freesurfer_subject_dir))
    return np.linalg.inv(brain_image.affine) @ c_ras_affine


def build_image_address(
    white_coordinates: np.ndarray,
    pial_coordinates: np.ndarray,
    faces: np.ndarray,
    image_shape: tuple[int, int, int],
    surface_to_voxel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return voxel indices and prism barycentric coordinates for a cortex hemisphere."""
    white_voxel = apply_affine(surface_to_voxel, white_coordinates)
    pial_voxel = apply_affine(surface_to_voxel, pial_coordinates)

    white_face_coordinates = np.transpose(white_voxel[:, faces], (1, 0, 2))
    pial_face_coordinates = np.transpose(pial_voxel[:, faces], (1, 0, 2))

    stacked_faces = np.vstack([white_face_coordinates, pial_face_coordinates])
    minimum_coordinates = np.min(stacked_faces, axis=0)
    maximum_coordinates = np.max(stacked_faces, axis=0)
    minimum_coordinates[minimum_coordinates < 0] = 0

    image_shape_array = np.reshape(np.asarray(image_shape, dtype=np.int32), (3, 1))
    for axis_index, axis_size in enumerate(image_shape_array):
        maximum_coordinates[axis_index, maximum_coordinates[axis_index] >= axis_size] = axis_size - 1

    minimum_coordinates = np.ceil(minimum_coordinates)
    maximum_coordinates = np.floor(maximum_coordinates) + 1
    dimensions = maximum_coordinates - minimum_coordinates
    dimensions[dimensions < 0] = 0

    voxel_count_per_face = np.prod(dimensions, axis=0).astype(np.int64)
    valid_face_indices = np.where(voxel_count_per_face > 0)[0]
    if valid_face_indices.size == 0:
        return {
            "coordinates": np.zeros((3, 0), dtype=np.float64),
            "faces": np.zeros((3, 0), dtype=np.int32),
            "voxel_indices": np.zeros((3, 0), dtype=np.int32),
        }

    face_end_offsets = np.cumsum(voxel_count_per_face[valid_face_indices])
    face_start_offsets = np.concatenate([[0], face_end_offsets[:-1]])
    total_voxel_candidates = int(face_end_offsets[-1])

    voxel_indices = np.zeros((3, total_voxel_candidates), dtype=np.int32)
    white_face_stack = np.zeros((3, 3, total_voxel_candidates), dtype=np.float64)
    pial_face_stack = np.zeros((3, 3, total_voxel_candidates), dtype=np.float64)
    original_face_indices = np.zeros(total_voxel_candidates, dtype=np.int32)

    active_faces = valid_face_indices
    query_offset = 0
    active_minimum = minimum_coordinates[:, valid_face_indices]
    active_dimensions = dimensions[:, valid_face_indices]
    active_voxel_counts = voxel_count_per_face[valid_face_indices]
    active_starts = face_start_offsets.astype(np.int64)

    while active_faces.size > 0:
        white_face_stack[:, :, active_starts] = white_face_coordinates[:, :, active_faces]
        pial_face_stack[:, :, active_starts] = pial_face_coordinates[:, :, active_faces]
        original_face_indices[active_starts] = active_faces
        voxel_indices[:, active_starts] = [
            (query_offset // (active_dimensions[1] * active_dimensions[2])) + active_minimum[0],
            np.mod(query_offset // active_dimensions[2], active_dimensions[1]) + active_minimum[1],
            np.mod(query_offset, active_dimensions[2]) + active_minimum[2],
        ]

        query_offset += 1
        still_active = np.where(active_voxel_counts > query_offset)[0]
        active_voxel_counts = active_voxel_counts[still_active]
        active_faces = active_faces[still_active]
        if active_faces.size == 0:
            break
        active_minimum = active_minimum[:, still_active]
        active_dimensions = active_dimensions[:, still_active]
        active_starts = active_starts[still_active] + 1

    coordinates = prism_barycentric_coordinates(white_face_stack, pial_face_stack, voxel_indices)
    inside_prism = ~np.isclose(np.sum(coordinates, axis=0), 0.0)
    return {
        "coordinates": coordinates[:, inside_prism],
        "faces": faces[:, original_face_indices[inside_prism]],
        "voxel_indices": voxel_indices[:, inside_prism],
    }


def interpolate_face_values(address: dict[str, np.ndarray], surface_data: np.ndarray, method: str) -> np.ndarray:
    """Interpolate per-vertex surface data at prism-addressed voxels."""
    coordinates = address["coordinates"]
    faces = address["faces"]
    if coordinates.size == 0:
        return np.zeros((0,), dtype=np.float64)

    weight_a = coordinates[0]
    weight_b = coordinates[1]
    weight_c = 1.0 - weight_a - weight_b
    triangle_values = surface_data[faces]

    if method == "nearest":
        winning_vertex = np.argmax(np.vstack([weight_a, weight_b, weight_c]), axis=0)
        return triangle_values[winning_vertex, np.arange(triangle_values.shape[1])]

    return triangle_values[0] * weight_a + triangle_values[1] * weight_b + triangle_values[2] * weight_c


def make_output_image(data: np.ndarray, template_image: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Create a NIfTI image aligned to the FreeSurfer template image."""
    header = nib.Nifti1Header()
    header.set_data_shape(data.shape)
    header.set_zooms(template_image.header.get_zooms()[:3])
    output_image = nib.Nifti1Image(data, template_image.affine, header=header)
    output_image.set_qform(template_image.affine, code=1)
    output_image.set_sform(template_image.affine, code=1)
    return output_image


def project_packaged_surfaces_to_volume(
    freesurfer_subject_dir: Path,
    prf_surface_dir: Path,
    output_dir: Path,
    metrics: tuple[str, ...] = ("polarAngle", "eccentricity", "rfWidth", "varea", "r2"),
    native_surface_dir: Path | None = None,
) -> list[Path]:
    """Project packaged native surface predictions to volume images."""
    brain_image_path = freesurfer_subject_dir / "mri" / "brain.mgz"
    if not brain_image_path.exists():
        raise FileNotFoundError(f"Missing brain image: {brain_image_path}")

    brain_image = nib.load(str(brain_image_path))
    image_shape = brain_image.shape[:3]

    hemisphere_addresses: dict[str, dict[str, np.ndarray]] = {}
    for hemisphere in ("lh", "rh"):
        white_coordinates, pial_coordinates, faces, coordinates_are_scanner_ras = load_surface_pair(
            freesurfer_subject_dir=freesurfer_subject_dir,
            hemisphere=hemisphere,
            native_surface_dir=native_surface_dir,
        )
        surface_to_voxel = build_surface_to_voxel_affine(
            brain_image=brain_image,
            freesurfer_subject_dir=freesurfer_subject_dir,
            coordinates_are_scanner_ras=coordinates_are_scanner_ras,
        )
        hemisphere_addresses[hemisphere] = build_image_address(
            white_coordinates.T.astype(np.float64),
            pial_coordinates.T.astype(np.float64),
            faces.T.astype(np.int32),
            image_shape,
            surface_to_voxel,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for metric_name in metrics:
        if metric_name not in METRIC_METHODS:
            raise ValueError(f"Unsupported metric: {metric_name}")

        output_dtype = np.int32 if metric_name == "varea" else np.float32
        output_volume = np.zeros(image_shape, dtype=output_dtype)
        for hemisphere in ("lh", "rh"):
            metric_path = prf_surface_dir / f"{hemisphere}.{metric_name}"
            surface_data = load_morph_or_gifti(metric_path)
            interpolated_values = interpolate_face_values(
                hemisphere_addresses[hemisphere],
                surface_data,
                method=METRIC_METHODS[metric_name],
            )
            if metric_name == "varea":
                interpolated_values = np.rint(interpolated_values).astype(np.int32)
            else:
                interpolated_values = interpolated_values.astype(np.float32)

            voxel_indices = hemisphere_addresses[hemisphere]["voxel_indices"]
            output_volume[tuple(voxel_indices)] = interpolated_values

        output_path = output_dir / f"{metric_name}.nii.gz"
        nib.save(make_output_image(output_volume, brain_image), str(output_path))
        written_paths.append(output_path)
        print(f"  wrote {output_path.name}")

    return written_paths


def project_surface_metric_to_volume(
    freesurfer_subject_dir: Path,
    surface_metric_path: Path,
    output_path: Path,
    hemisphere: str,
    method: str = "linear",
    native_surface_dir: Path | None = None,
) -> Path:
    """Project one native surface metric into a brain.mgz-aligned volume."""
    brain_image_path = freesurfer_subject_dir / "mri" / "brain.mgz"
    if not brain_image_path.exists():
        raise FileNotFoundError(f"Missing brain image: {brain_image_path}")
    if method not in {"linear", "nearest"}:
        raise ValueError(f"Unsupported interpolation method: {method}")

    brain_image = nib.load(str(brain_image_path))
    white_coordinates, pial_coordinates, faces, coordinates_are_scanner_ras = load_surface_pair(
        freesurfer_subject_dir=freesurfer_subject_dir,
        hemisphere=hemisphere,
        native_surface_dir=native_surface_dir,
    )
    surface_to_voxel = build_surface_to_voxel_affine(
        brain_image=brain_image,
        freesurfer_subject_dir=freesurfer_subject_dir,
        coordinates_are_scanner_ras=coordinates_are_scanner_ras,
    )
    address = build_image_address(
        white_coordinates.T.astype(np.float64),
        pial_coordinates.T.astype(np.float64),
        faces.T.astype(np.int32),
        brain_image.shape[:3],
        surface_to_voxel,
    )

    surface_data = load_morph_or_gifti(surface_metric_path)
    interpolated_values = interpolate_face_values(address, surface_data, method=method).astype(np.float32)
    output_volume = np.zeros(brain_image.shape[:3], dtype=np.float32)
    output_volume[tuple(address["voxel_indices"])] = interpolated_values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(make_output_image(output_volume, brain_image), str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freesurfer-subject-dir", required=True, help="FreeSurfer subject output directory containing mri/ and surf/.")
    parser.add_argument("--native-surface-dir", help="Optional directory with c_ras-aligned native GIFTI white/pial surfaces.")
    parser.add_argument("--prf-surface-dir", help="Directory containing packaged pRF surface files such as lh.polarAngle.")
    parser.add_argument("--output-dir", help="Directory to write projected .nii.gz volumes.")
    parser.add_argument("--surface-metric", help="Single native surface metric to project.")
    parser.add_argument("--hemisphere", choices=["lh", "rh"], help="Hemisphere for --surface-metric.")
    parser.add_argument("--output-volume", help="Output path for --surface-metric projection.")
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear", help="Interpolation for --surface-metric.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["polarAngle", "eccentricity", "rfWidth", "varea", "r2"],
        help="Metrics to project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native_surface_dir = Path(args.native_surface_dir) if args.native_surface_dir else None

    if args.surface_metric:
        if not args.hemisphere or not args.output_volume:
            raise ValueError("--surface-metric requires --hemisphere and --output-volume")
        output_path = project_surface_metric_to_volume(
            freesurfer_subject_dir=Path(args.freesurfer_subject_dir),
            surface_metric_path=Path(args.surface_metric),
            output_path=Path(args.output_volume),
            hemisphere=args.hemisphere,
            method=args.method,
            native_surface_dir=native_surface_dir,
        )
        print(output_path)
        return

    if not args.prf_surface_dir or not args.output_dir:
        raise ValueError("Packaged projection requires --prf-surface-dir and --output-dir")

    output_paths = project_packaged_surfaces_to_volume(
        freesurfer_subject_dir=Path(args.freesurfer_subject_dir),
        prf_surface_dir=Path(args.prf_surface_dir),
        output_dir=Path(args.output_dir),
        metrics=tuple(args.metrics),
        native_surface_dir=native_surface_dir,
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
