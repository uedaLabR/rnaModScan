# HeLa candidate composition

Completed HeLa source tables provide the motif and repeat composition of GLORI/Slic-seq, depth/AF-qualified Dorado/modkit candidates before NN filtering, and raw NN candidates before reference rescue. This script does not read BAM files or run an NN.

```bash
MPLCONFIGDIR=/tmp/rnamodscan-mpl .venv310/bin/python analysis/hela_composition/build.py
```

Default outputs are new `paper/data/hela_composition_20261109` and `paper/figures/hela_composition_20261109` directories. Existing outputs are not overwritten; specify different `--data` and `--figures` paths when reproducing the analysis. The workbook has three sheets with every plotted count, exact numeric proportion, denominator and source note. Repeat counts are independently reaggregated from the existing 75,007-site membership table and compared with the historical summary. The Excel writer checks native numeric values after read-back.

One Dorado m6A site has unknown sequence context. It remains in the total-candidate count but is excluded from the DRACH/non-DRACH proportion denominator. Any Alu is a subset of any repeat, not a mutually exclusive category. These figures describe candidate composition; they are not an independent accuracy estimate or a background-controlled enrichment test.

The two-panel composition figure and separate C/D panels are saved as SVG, PDF and PNG. `Fig4C_D_candidate_composition` contains C/D labels for appending below the original Venn drawing; its caveats belong in the Word caption. Existing Venn diagrams and source files are preserved.

The original three-sheet composition workbook remains unchanged under `archive_initial_three_sheet/`. The active four-sheet workbook, `HeLa_composition_and_resources_20261109.xlsx`, adds `ResourceTableS18` from `paper/data/runtime_comments_20261109/runtime_table.tsv`, including native numeric runtime, process RAM and per-process GPU VRAM values. CPU full-workflow time is explicitly estimated; its full-workflow memory is unmeasured. GPU process memory observations use a different method from the CPU GNU-time high-water measure, as recorded in the resource-table columns and evidence JSON. Exclude `archive_*` folders when creating the current manuscript bundle.
