# Reference source data

RMBase 3.0 data originate from the
[RMBase download page](https://rna.sysu.edu.cn/rmbase3/download.php).
`SHA256SUMS` records the original archives, including human files that must be
obtained separately. The importer reads archive members without extracting
them into the project tree and retains source provenance.

## Files included in Git

The four mouse source archives are intentionally retained for reference
preparation after cloning:

| Genome | Modification | Archive | Size (MiB) |
|---|---|---|---:|
| mm10 | m6A | `mm10.m6A.tar.gz` | 43.00 |
| mm10 | m5C | `mm10.m5C.tar.gz` | 1.95 |
| mm10 | Inosine | `mm10.RNA-editing.tar.gz` | 0.43 |
| mm10 | Pseudouridine | `mm10.Pseudo.tar.gz` | 0.29 |

From the repository root, run once:

```bash
python analysis/reference_catalog/prepare_mouse_reference_catalog.py
```

The generated BEDs and provenance TSVs are stored in
`data/reference_sites_mm10/rmbase3/` and excluded from Git. The small catalog is
`data/references/catalog.mm10.json`. Matching completed imports are verified and
reused. RNA-editing entries are restricted to `A-I` for inosine. Reference site
counts describe catalog entries, not modification abundance in a sample.

## Human reference data

All `hg38.*.tar.gz` downloads and imported `references/rmbase3/` data are excluded
from Git. Their local copies are preserved. Obtain the following files from
RMBase 3.0 and place them under `source/` when preparing human references:

| Modification | Archive | Size (MiB) |
|---|---|---:|
| m6A | `hg38.m6A.tar.gz` | 59.65 |
| m5C | `hg38.m5C.tar.gz` | 2.22 |
| Inosine | `hg38.RNA-editing.tar.gz` | 7.25 |
| Pseudouridine | `hg38.Pseudo.tar.gz` | 0.35 |

For a database-only human catalog, use a separate, ignored output location:

```python
from pathlib import Path
from utils.references.rmbase import prepare_rmbase3_catalog

prepare_rmbase3_catalog(
    Path('source'),
    Path('references/rmbase3'),
    Path('configs/local/catalog.hg38.rmbase.json'),
    genome='hg38',
)
```

Set `references.catalog` to that JSON file. This catalog enables RMBase overlap;
it does not add experimental short-read references. The included
`data/references/catalog.json` describes the original full human catalog and
requires all listed experimental files and provenance as well. These are not
bundled or downloaded automatically. Their assembly, cell, method, source paths
and checksums must match the catalog. See the
[reference preparation guide](../analysis/reference_catalog/README.md).

Reference files use BED0 genomic coordinates. No automatic liftover or chromosome
renaming is performed. A matching softmasked FASTA with FAI is supplied separately.
Third-party datasets retain their original terms and attribution.

## Runtime files and import audits

`source/hg38.m6A.tar.gz` is the original human RMBase m6A download (59.65 MiB).
It is needed to regenerate the imported reference, not to run prediction with an
already prepared BED. Like the other human source archives, it is obtained from
the RMBase download page and checked against `source/SHA256SUMS`.

Prediction reads the BEDs named by the selected catalog and the associated
provenance JSON files, including `references/rmbase3/import.json`. The detailed
`references/rmbase3/*.provenance.tsv` import audits are not read by the prediction
runtime and remain excluded from Git. Preserve them locally when retaining the
complete import record.
