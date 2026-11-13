# Understand the output flags

rnaModScan combines filtering with a neural network classifier and separate
reference- and sequence-based retention rules. The flags record which rules
retained an **original modification candidate**; they are not validation labels
or estimates of chemical accuracy.

## From candidates to the filtered BED

Candidates first pass the configured depth, modified-read-count and modification
fraction (AF) thresholds. The supplied templates use depth ≥10, modified reads
≥3 and AF ≥0.10. A reference or motif match does not bypass these initial
requirements. See the `candidates` settings in
[`prediction/config.yaml`](../prediction/config.yaml).

For m6A and inosine, the neural network classifier assigns scores to m6A,
inosine and false-positive classes. It supports retention when the
highest-scoring class matches the original candidate's modification. Database,
experimental-reference and m6A motif rules can retain candidates independently
of this decision. m5C and pseudouridine (`pseU`) use reference overlap only.

The main output is
`<output_dir>/predictions/bedmethyl/filtered.bed`. Target-specific copies are
written to `predictions/bedmethyl/<target>/filtered.bed`. Both retain the
original input columns in their original order, including the original counts,
reported percentage and any additional source columns, then append the five
columns below. **Only retained rows are exported.** The workflow does not add
sites absent from the original candidates or create an m6A row from an
inosine-only call.

## Five added columns

| Column | Values | Meaning |
|---|---|---|
| `flag_db` | `0` or `1` | `1` when an original WT candidate matches an applicable database entry, such as RMBase, for the same modification, chromosome, position and strand. |
| `flag_short_read` | `0` or `1` | `1` when an original WT candidate matches a published experimental site in the applicable reference-cell scope, such as GLORI for m6A or Slic-seq for inosine. |
| `flag_nn` | `0` or `1` | `1` when the neural network classifier's highest-scoring class matches the original m6A or inosine candidate. It is `0` for reference-only targets. |
| `motif_keep` | `0`, `1` or `2` | A code for the enabled m6A motif rule that retained the candidate; see below. |
| `filter_pass` | `1` in the exported BED | At least one of the database, experimental-reference, classifier or two motif rules retained the candidate. |

The `flag_short_read` name is retained for compatibility. Its catalog category
also contains legacy PUS7 pseudouridine reference records derived from nanopore
data, with explicit source-cell mappings. Consult the recorded source IDs and
[`catalog.json`](../data/references/hg38/catalog.json) for the actual assay and
cell scope. The flag does not mean that every experimental source used
short-read sequencing.

Database and experimental-reference flags apply only to WT samples. Their
availability also depends on the selected genome, reference catalog and
reference cell. There is no coordinate liftover or chromosome renaming.

### Motif codes

| `motif_keep` | Applied retention rule |
|---|---|
| `0` | Neither enabled motif rule applied. This can mean no motif match, a disabled rule, or a target other than m6A. It does **not** imply rejection. |
| `1` | The original m6A candidate is in a DRACH motif and `filtering.m6a_drach_keep` is enabled. |
| `2` | The original m6A candidate matches canonical RNA `UAC[m6A]GAGAA` and `filtering.m6a_mettl16_keep` is enabled. |

DRACH means D={A,G,U}, R={A,G}, A, C and H={A,C,U}, with the central
adenine as the candidate. The canonical METTL16-associated motif is exactly
`UAC[m6A]GAGAA`; the candidate adenine is the fourth base of this nine-base
sequence. Sequence matching uses the candidate's strand orientation. Genome
FASTA uses T in place of RNA U. The broader `UAC[m6A]GARAA` pattern is not used.
The two motif rules cannot both match the same candidate.

The prediction and Docker templates enable both motif rules. Setting either key
to `false` disables that rule. An omitted key defaults to `false`, including
`m6a_mettl16_keep` in older configurations. Both rules apply to qualifying m6A
candidates in WT and IVT samples, even if the classifier rejects them. A motif
code records an applied retention rule, not a complete annotation of all motif
matches or evidence of a particular writer enzyme.

## Combine the evidence

The detailed decision TSVs record five independent binary flags:
`flag_db`, `flag_short_read`, `flag_nn`, `flag_drach` and `flag_mettl16`.
The BED combines the last two into `motif_keep` to keep the output compact.
The final decision is:

```text
filter_pass = flag_db OR flag_short_read OR flag_nn OR flag_drach OR flag_mettl16
```

Several flags can be `1` for the same candidate; these categories are not
exclusive. A zero means the corresponding retention evidence was not applied,
which can include an unavailable reference, a disabled rule or an unsuccessful
classifier decision. It is not proof that the site lacks the modification.
Rejected candidates have `filter_pass=0` in the detailed decision TSVs and are
absent from the filtered BED. The TSVs also record the matching reference source
IDs and classifier scores.

## Small example

These are **illustrative rows**, not analysis results. Each has 18 original
modkit columns followed by the five added columns. Both original m6A candidates
have ten reads, three modified reads and AF=0.30. Assume the second candidate
matches the enabled canonical METTL16 motif.

```text
# rnaModScan filtered bedMethyl: original input columns are retained in their original order.
# appended_columns	flag_db	flag_short_read	flag_nn	motif_keep	filter_pass
# flag_db, flag_short_read, flag_nn: 0=absent, 1=present; filter_pass=1 for every exported row.
# motif_keep: 0=none; 1=DRACH; 2=canonical METTL16 UAC[m6A]GAGAA (candidate A is nonamer position 4).
chr1	100	101	a	10	+	100	101	0,0,0	10	30.00	3	7	0	0	0	0	0	1	0	1	0	1
chr1	200	201	a	10	+	200	201	0,0,0	10	30.00	3	7	0	0	0	0	0	0	0	0	2	1
```

The first row passes through both the database and classifier flags, despite
`motif_keep=0`. The second passes through the canonical METTL16 motif rule,
despite all three other BED evidence flags being zero. Header comments appear
in combined and target-specific BEDs, including files with no retained rows.

## Optional BAM output

Use `--recalibrate` to write a new BAM with rejected target modification
confidence values set to zero. Candidates retained through any rule, including
the motif rules, remain retained in that output. The original BAM is preserved.
The BED flags describe site-level retention; they do not change the original
bedMethyl counts or AF. See the [prediction guide](../prediction/README.md) for
commands and output paths.
