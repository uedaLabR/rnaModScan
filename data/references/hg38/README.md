# Included human prediction references

This is the portable GRCh38/hg38 reference bundle used by default for human
prediction. It contains 17 losslessly compressed BEDs, three original provenance
JSON files and `catalog.json` (19 source entries). Runtime data total about
12.3 MiB; the largest file is about 6.8 MiB. Every included file is below 50 MiB.

```text
catalog.json       Portable source catalog; all paths are relative to this directory
SHA256SUMS         Checksums of the catalog, compressed BEDs and provenance JSON
by_method/         HEK293T/HeLa GLORI, Slic-seq, BACS, m5C-TAC-seq, BID-seq and PRAISE BEDs
sources.json       Original experimental-map provenance: papers, source hashes and conversions
rmbase3/           RMBase 3.0 m6A, inosine, m5C and pseudouridine BEDs, plus import.json
pus7/              Two existing PUS7 evidence BEDs and their original provenance.json
```

## Use and verify

The normal prediction configuration selects this bundle with:

```yaml
genome: hg38
references:
  fasta: /path/to/matching/softmasked/hg38.fa
  catalog: null
```

The loader opens `.bed.gz` directly; do not decompress or edit the included files.
From this directory, check every runtime file:

```bash
sha256sum -c SHA256SUMS
```

You must still supply a matching softmasked genome FASTA/FAI and your own sample
BAM and pileup inputs. This human catalog is not selected for `mm10` or `other`.
An explicit `references.catalog` path overrides the default for the applicable
genome; the original local catalog remains at `../catalog.json` for historical
analyses.

## Contents and provenance

Each BED decompresses to the exact bytes of its original prepared file. Source
IDs, assembly, BED0 coordinates, strand, modification, cell scopes and metadata
are unchanged. Every catalog entry records the compressed-file `sha256` and
original `uncompressed_sha256`. The three provenance JSON files are byte-identical
copies of the original records. Historical source/FASTA paths in those records
describe the import; the original machines and raw files are not needed at
prediction time.

RMBase 3.0 applies across human biosamples and is not a cell-specific truth set.
The experimental maps use the explicitly configured HEK293T/HeLa scopes. The
PUS7 evidence is the existing, explicitly configured HepG2-derived Nanopore
evidence applied to HEK293T/HeLa; its legacy `short_read` bucket is a reporting
category, not a claim that the assay used short reads. It remains excluded from
the manuscript's baseline short-read unions. See `pus7/provenance.json` for its
scope and limitations.

Original source archives, detailed `*.provenance.tsv` import audits, genome
FASTA/FAI, training tensors and analysis outputs are not part of this bundle.
The source papers and download locations are recorded in the JSON files. These
third-party datasets retain their original terms and attribution; the repository's
software license does not replace the source datasets' terms.
