# Third-party reference data and attribution

The root [MIT License](LICENSE) covers rnaModScan software. It does not replace
the terms of the reference datasets, external tools, libraries, or base images.
The software metadata's `MIT` identifier must not be read as permission to
redistribute every bundled dataset under MIT.

The table below identifies the reference data already tracked in the repository.
The preparation review was performed on 26 November 2026. Publisher article
licenses are evidence about the article; they are not automatically a separate
license grant for every deposited dataset. Dataset-specific terms and any
third-party exclusions still need to be checked before publishing a release.

| Source | Included files | Citation and original source | Terms found / unresolved checks |
| --- | --- | --- | --- |
| RMBase 3.0 | `data/references/hg38/rmbase3/*.bed.gz`; four `source/mm10.*.tar.gz` archives | Xuan et al., *RMBase v3.0: decode the landscape, mechanisms and functions of RNA modifications*, [doi:10.1093/nar/gkad1070](https://doi.org/10.1093/nar/gkad1070); [download page](https://rna.sysu.edu.cn/rmbase3/download.php) | The paper states that data use follows the GNU Public License and primary-source licenses. The GNU license version and applicable primary-source redistribution conditions are not specified in the repository. Confirm these and preserve required notices before release. |
| GLORI | `by_method/*_m6A_GLORI_GRCh38.bed.gz` | Liu et al., *Absolute quantification of single-base m6A methylation in the mammalian transcriptome using GLORI*, [doi:10.1038/s41587-022-01487-9](https://doi.org/10.1038/s41587-022-01487-9); GEO sample accessions and source hashes in `sources.json` | Separate dataset redistribution terms have not been verified. |
| Slic-seq | `by_method/*_inosine_Slic-seq_GRCh38.bed.gz` | [doi:10.1093/nar/gkad604](https://doi.org/10.1093/nar/gkad604); `GSE169710_Data_S2_A-to-I_sites_identified_by_slic-seq.xlsx` | The article carries a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) notice. Confirm the terms for the deposited workbook and derived BEDs; do not label them MIT. |
| BACS | `by_method/HeLa_*_BACS_GRCh38.bed.gz` | Xu et al., *Absolute quantitative and base-resolution sequencing reveals comprehensive landscape of pseudouridine across the human transcriptome*, [doi:10.1038/s41592-024-02439-8](https://doi.org/10.1038/s41592-024-02439-8); `GSE241849_Supplementary_Table_3_10.xlsx` | Publisher article notice: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with third-party-material exclusions. Retain attribution and transformation details; confirm that the deposited workbook is covered. |
| m5C-TAC-seq | `by_method/*_m5C_m5C-TAC-seq_GRCh38.bed.gz` | Lu et al., *Base-resolution m5C profiling across the mammalian transcriptome by bisulfite-free enzyme-assisted chemical labeling approach*, [doi:10.1016/j.molcel.2024.06.021](https://doi.org/10.1016/j.molcel.2024.06.021) | Separate dataset redistribution terms have not been verified. Historical liftover source files and hashes are recorded in `sources.json`. |
| BID-seq | `by_method/*_psi_BID-seq_GRCh38.bed.gz` | Dai et al., *Quantitative sequencing using BID-seq uncovers abundant pseudouridines in mammalian mRNA at base resolution*, [doi:10.1038/s41587-022-01505-w](https://doi.org/10.1038/s41587-022-01505-w); `GSE179798_HeLa_mRNA_WT_BID-seq.xlsx` and HEK293T provenance in `sources.json` | Publisher article notice: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with third-party-material exclusions. Confirm coverage of the specific deposited maps. |
| PRAISE | `by_method/HEK293T_psi_PRAISE_GRCh38.bed.gz` | Zhang et al., *Quantitative profiling of pseudouridylation landscape in the human transcriptome*, [doi:10.1038/s41589-023-01304-7](https://doi.org/10.1038/s41589-023-01304-7) | Separate dataset redistribution terms have not been verified. |
| PUS7 | `pus7/S1A.bed.gz`, `pus7/S1B.bed.gz` | Rodell et al., *PUS7 mRNA pseudouridylation is driven by RNA sequence, structure, and cell-type-specific features*, [doi:10.1016/j.xgen.2026.101332](https://doi.org/10.1016/j.xgen.2026.101332), [article license notice](https://pmc.ncbi.nlm.nih.gov/articles/PMC13689517/); source `mmc2.xlsx` and SHA-256 in `pus7/provenance.json` | The article carries [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Permission for redistributing the extracted/converted BEDs is not established by this review. Confirm the dataset terms or obtain permission before release. |

`by_method/` and `pus7/` paths in this table are relative to
`data/references/hg38/`. Experimental-map provenance, source filenames, original
hashes and coordinate conversions are retained in
[`sources.json`](data/references/hg38/sources.json) and
[`pus7/provenance.json`](data/references/hg38/pus7/provenance.json).
RMBase import provenance is in
[`rmbase3/import.json`](data/references/hg38/rmbase3/import.json).
The prepared BEDs use BED0 coordinates and lossless gzip compression. Some
source positions were converted from one-based coordinates or lifted to GRCh38
as recorded in the original provenance. No reference coordinates, cell scopes,
content, or checksums were changed during release preparation.

The four mouse archives are original RMBase downloads with checksums in
[`source/SHA256SUMS`](source/SHA256SUMS). They are used for mouse reference
preparation and are unnecessary for the manuscript's human-only analyses, but
remain included to preserve the documented mouse setup. Their inclusion needs
the same upstream license review as the human RMBase maps.

Genome FASTA/FAI, GENCODE annotation, raw BAM/POD5/FAST5 reads, training tensors
and original human source archives are not supplied in the Git archive.
Users obtain these separately under their providers' terms. Dependency files
and Dockerfiles specify external packages and images; their code and container
layers are not vendored in the source archive. Dorado and modkit executables are
not bundled.

**Release status:** reference-data redistribution clearance is incomplete.
Keep this distinct from metadata/schema validation and archive-integrity
checks. This preparation does not grant permissions on behalf of upstream
rights holders, and no GitHub Release or Zenodo record has been published.
