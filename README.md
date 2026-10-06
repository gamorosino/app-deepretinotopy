[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-brainlife.app.6ac02d2c931afedae84f7da2-blue.svg)](https://brainlife.io/app/6ac02d2c931afedae84f7da2)

# deepRetinotopy

`app-deepretinotopy`

This app predicts retinotopic maps (polar angle, eccentricity, and population receptive field size) directly from an individual's cortical anatomy, using the geometric deep learning model from [deepRetinotopy_TheToolbox](https://github.com/felenitaribeiro/deepRetinotopy_TheToolbox).

Only a minimal set of FreeSurfer surfaces is required per subject -- no functional/retinotopic mapping scan is needed.

---

# Author

**Gabriele Amorosino** ([gabriele.amorosino@utexas.edu](mailto:gabriele.amorosino@utexas.edu))

Wraps the [deepRetinotopy_TheToolbox](https://github.com/felenitaribeiro/deepRetinotopy_TheToolbox) by Fernanda L. Ribeiro et al. for use on [brainlife.io](https://brainlife.io).

---

# Contributors

**Junbeom Kwon** ([junebeomstics@utexas.edu](mailto:junebeomstics@utexas.edu))

---

# Citation

If you use this app in your research, please cite:

## Primary work

Ribeiro, F. L., Bollmann, S., & Puckett, A. M. (2021). Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning. NeuroImage, 118624.
[https://doi.org/10.1016/j.neuroimage.2021.118624](https://doi.org/10.1016/j.neuroimage.2021.118624)

Ribeiro, F. L., et al. (2025). Predicting functional topography of the human visual cortex from cortical anatomy at scale. bioRxiv.
[https://doi.org/10.1101/2025.11.27.690210](https://doi.org/10.1101/2025.11.27.690210)

## Software dependencies

**deepRetinotopy_TheToolbox / Neurodesk container** (`vnmd/deepretinotopy_1.0.19`)

Renton, A. I., et al. (2024). Neurodesk: an accessible, flexible and portable data analysis environment for reproducible neuroimaging. Nature Methods, 21(5), 804-808.
[https://doi.org/10.1038/s41592-023-02145-x](https://doi.org/10.1038/s41592-023-02145-x)

**Brainlife.io**

Hayashi, S., et al. (2024). Brainlife.io: a decentralized platform for reproducible neuroscience. Nature Methods.
[https://doi.org/10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)

---

# Overview

The app runs the full deepRetinotopy pipeline on a single subject:

1. **Surface generation** -- midthickness surface + curvature computed from native `white`/`pial`, resampled onto the HCP 32k `fs_LR` mesh.
2. **Prediction** -- a pretrained geometric deep learning model predicts polar angle, eccentricity, and/or pRF size maps from curvature.
3. **Resampling** -- predicted maps are resampled back to the subject's native surface space.

---

# Inputs

| Key           | Type   | Description                                                                 |
| ------------- | ------ | ---------------------------------------------------------------------------- |
| `freesurfer`  | dir    | Path to a **single subject's** FreeSurfer output directory (required)        |
| `maps`        | string | Comma-separated maps to predict (default: `polarAngle,eccentricity,pRFsize`) |
| `dataset`     | string | Dataset name label, passed through to the toolbox (default: `brainlife`)     |
| `subject_id`  | string | Internal subject id used when staging input for the toolbox (default: `subject`) |
| `fast`        | string | `yes`/`no`, fast midthickness generation (default: `yes`)                    |
| `use_gpu`     | bool   | Pass `--nv` to Singularity for GPU inference (default: `false`)              |
| `container_version` | string | `vnmd/deepretinotopy_<version>` tag to pull (default: `1.0.18`)       |

**Note on `container_version`:** `1.0.19` fails to build via Singularity/Apptainer on older packaged installs (confirmed broken on `singularity-ce 4.1.1`, Ubuntu Noble's packaged build) -- extraction breaks unpacking a hardlinked file from a conda package (`brotlicffi`), in both rootless and `--fakeroot` mode, even with a fresh `TMPDIR`/`CACHEDIR`. This is a bug in the older Singularity/Apptainer OCI-to-SIF conversion path, not a broken image: `docker pull`/`docker run` of `1.0.19` works fine, and so does `singularity exec`/`apptainer exec` with a current Apptainer (confirmed working on Apptainer `1.5.3`). `1.0.18` is the default here for maximum compatibility across execution environments with an unknown Singularity/Apptainer version; set `container_version` to `1.0.19` explicitly if your environment has a reasonably recent Apptainer/SingularityCE.

Minimal expected FreeSurfer input:

```
freesurfer/
└── surf/
    ├── lh.white
    ├── lh.pial
    ├── lh.sphere.reg
    ├── rh.white
    ├── rh.pial
    └── rh.sphere.reg
```

No FreeSurfer license is required for this app (only `mris_convert`/`wb_command` are used, not `recon-all`).

The HCP `fs_LR-deformed_to-fsaverage` template surfaces required by the toolbox (`-t`) are bundled in [templates/](templates/) -- fetched from [Washington-University/HCPpipelines](https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage), no additional input needed.

---

# Output

```
output/
└── subject/
    ├── surf/               # midthickness + curvature (32k fs_LR)
    └── deepRetinotopy/      # fs_predicted_* (32k fs_LR) and *.native.func.gii maps
prf_surfaces/
├── lh.polarAngle.native.func.gii
├── rh.polarAngle.native.func.gii
├── lh.eccentricity.native.func.gii
├── rh.eccentricity.native.func.gii
├── lh.rfWidth.native.func.gii
└── rh.rfWidth.native.func.gii
product.json
```

`prf_surfaces/` flattens the native-space predictions with naming consistent with this ecosystem's other retinotopy apps (`rfWidth` = pRF size), so they can be consumed directly by downstream apps such as `app-retinotopic-connectivity`. Note deepRetinotopy does not produce a visual-area (`varea`) parcellation.

---

# Requirements

* [Singularity](https://sylabs.io/singularity/) or Apptainer
* Docker image: `vnmd/deepretinotopy_1.0.19` (pulled automatically via `docker://`)
* GPU inference (optional): CUDA 12.4-compatible GPU (H100, A100, L40), pass `use_gpu: true` / `--use_gpu`

---

# Running the App

### On Brainlife.io

Submit via the 'Execute' tab once registered.

### Running Locally

```bash
git clone <this-repo>
cd app-deepretinotopy

./main_cli.sh \
  --freesurfer /path/to/freesurfer/output \
  --maps "polarAngle,eccentricity,pRFsize" \
  --dataset hcp
```

Or, edit `config.json.sample`, copy it to `config.json`, and run:

```bash
./main
```

Set `NO_CONTAINER=1` to skip Singularity and run `deepRetinotopy` from a local install instead (e.g. Neurodesk `ml deepretinotopy/1.0.19`).

See `./main_cli.sh --help` for all options.

---

# License

GPL-3.0, matching the upstream [deepRetinotopy_TheToolbox](https://github.com/felenitaribeiro/deepRetinotopy_TheToolbox) license, since this app bundles its template surface files and wraps its container. See [LICENSE](LICENSE).
