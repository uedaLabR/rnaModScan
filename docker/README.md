# CPU and NVIDIA GPU prediction containers

The two Dockerfiles provide the same rnaModScan prediction workflow, deployment
weights and human reference BED/JSON bundle. They preserve TensorFlow 2.10.1 and
Keras 2.10.0, which are required to load the distributed saved model. No training
or separate model download is needed. These are **prediction images**; Dorado
basecalling, model training and manuscript tools are not included.

The commands below build images from a checkout;
they do not assume that prebuilt images are available in a registry.

## Platform and prerequisites

- Both recipes target **Linux containers on x86_64 (`linux/amd64`)**. A working
  Docker Engine or Docker Desktop is required. ARM-native images are not
  provided; ARM-host emulation has not been validated. The TensorFlow CPU binary
  requires a CPU with AVX instructions.
- The CPU image works without an NVIDIA GPU or host CUDA installation.
- The GPU image additionally needs a supported NVIDIA GPU, a compatible host
  driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  CUDA 11.2 and cuDNN 8 are supplied **inside the image**. A separate host CUDA
  toolkit is unnecessary. GPU-family support is limited by this TensorFlow/CUDA
  version. GPU passthrough must already work on the host; see the
  [TensorFlow Docker guide](https://www.tensorflow.org/install/docker).

The GPU recipe builds Python 3.10.22 in a separate stage on Ubuntu 20.04. Its
source archive is pinned by SHA256; the final image contains the interpreter and
runtime libraries, without the compiler stage. CPU and GPU base-image manifests
are pinned by digest. The CUDA/cuDNN combination follows TensorFlow's
[tested build configurations](https://www.tensorflow.org/install/source#gpu).

## Build and check

Run from the repository root:

```bash
docker build --platform linux/amd64 -f Dockerfile.cpu -t rnamodscan:cpu .
docker run --rm rnamodscan:cpu predict --help
docker run --rm --entrypoint python rnamodscan:cpu \
  /opt/rnaModScan/docker/smoke_test.py --device cpu

# NVIDIA hosts only. Building does not require a GPU; running with gpu does.
docker build --platform linux/amd64 -f Dockerfile.gpu -t rnamodscan:gpu .
docker run --rm --gpus all --entrypoint python rnamodscan:gpu \
  /opt/rnaModScan/docker/smoke_test.py --device gpu
```

Each Dockerfile runs `pip check`, CLI checks, bundle checksum checks and a
two-bag saved-model inference smoke test during its build. The explicit GPU
smoke command also fails if TensorFlow cannot use a GPU. Artificial smoke-test
inputs assess installation only and do not measure biological accuracy.

Validation status at preparation: the checks passed both in the existing
compatible Python environment and in a fresh temporary CPU virtual environment
installed from these requirements, using an isolated copy of the allowed build
context. `pip check` also passed. Docker builds and container GPU execution could
not be run on the preparation host because access to its Docker daemon was denied.
The image manifests and Python source checksum were verified against their
upstream endpoints. Run the commands above on a Docker-enabled host before using
the images for an analysis.

## Configure mounted inputs

Provide a sorted, indexed Dorado BAM, its modkit bedMethyl and matching pileup
log, and an assembly-matched **softmasked** genome FASTA with FAI. Genome and
sample data are not baked into either image. External `modkit` is not included:
the default configuration reuses prepared pileup files (`run_pileup: false`).
Produce those files using your existing modkit installation before container
prediction. For in-container pileup, extend the image with a compatible modkit
binary and its runtime dependencies, then enable `run_pileup` explicitly.

```bash
mkdir -p configs/local docker_output
cp docker/predict.yaml configs/local/docker-predict.yaml
```

Edit the copied YAML using paths **inside the container**, not host paths:

| Host content | Container path / setting |
|---|---|
| Input directory with BAM/index and pileup files | `/input` |
| Softmasked FASTA/FAI directory | `/genome`; template uses `/genome/hg38.fa` |
| Writable output directory | `/output`; choose a fresh `output_dir` per sample |
| Edited YAML | `/config/predict.yaml` |
| Included model | `/opt/rnaModScan/models/pooled_HEK293T_HeLa_20261111` |
| Included human catalog | `/opt/rnaModScan/data/references/hg38/catalog.json` |

Set `sample.sample_id`, `cell`, `reference_cell` and `condition` for your sample;
HEK293 and HEK293T are distinct. Update the three sample filenames and genome
path. Optional GTF annotation also uses a mounted container path. CPU/GPU images
contain the same model; their reported study performance is not a new evaluation
of the pooled deployment fit.

For a non-human reference, update `genome`, FASTA/FAI and reference catalog as
described in the main README. Prepared mouse references are not bundled into
these prediction images; mount an appropriate catalog and its files separately.

## Run prediction

Replace the two example absolute host directories below. The Linux `--user`
option makes output files belong to your host user. Input/genome/config mounts
are read-only; only the output mount is writable.

```bash
INPUT_DIR=/absolute/path/to/sample_inputs
GENOME_DIR=/absolute/path/to/softmasked_genome

docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$INPUT_DIR,target=/input,readonly" \
  --mount "type=bind,source=$GENOME_DIR,target=/genome,readonly" \
  --mount "type=bind,source=$PWD/configs/local/docker-predict.yaml,target=/config/predict.yaml,readonly" \
  --mount "type=bind,source=$PWD/docker_output,target=/output" \
  rnamodscan:cpu predict --config /config/predict.yaml --device cpu --dry-run
```

After inspecting the plan, run the same command with `--dry-run` replaced by
`--no-recalibrate`. For GPU inference, add `--gpus all` after `docker run --rm`,
use `rnamodscan:gpu`, and set `--device gpu`. Selecting `gpu` requires an available
GPU; `--device auto` permits a CPU fallback. On Docker Desktop, adapt the host-user
option to your host's file-sharing settings.

Output appears under `docker_output/analysis_sample/` with the template. The
principal BED is `predictions/bedmethyl/filtered.bed`; retained-site tables,
exclusion audits and summaries are also written. Set a new `output_dir` for each
full run. Optional `--recalibrate` writes a new BAM in the output directory;
the original read-only input BAM remains unchanged. Preparation/inference/filter
can also run separately using `--stage prepare`, `--stage infer` and
`--stage filter`; use one image consistently for one output tree.

## Build context

`.dockerignore` uses an allowlist, and Dockerfiles use explicit `COPY` paths.
Only prediction code, package metadata, container helpers, the frozen deployment
bundle and prepared human references enter the context. Local datasets, virtual
environments, training runs, manuscript files, Git state and credentials are
excluded. Reference data retain their original attribution and usage terms;
see `data/references/hg38/README.md`.
