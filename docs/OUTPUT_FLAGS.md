# Understand the output flags

rnaModScan combines filtering with a neural network classifier and separate
reference- and sequence-based retention rules. The flags record which rules
retained an **original modification candidate**; they are not validation labels
or estimates of chemical accuracy.

## From candidates to the filtered BED

Candidates first pass the configured depth, modified-read-count and modification
fraction (AF) thresholds. The supplied candidate defaults are depth ≥10,
modified reads ≥3 and AF ≥0.10, including the m5C/Ψ known-site route.
**Only m5C/Ψ motif retention (codes 3–6) adds depth ≥20 and AF ≥0.40**.
`filtering.auxiliary_motif_min_depth` and `filtering.auxiliary_motif_min_af`
configure this extra motif gate and default to 20 and 0.40 even when omitted.
The original candidate requirements still apply to every route; known-site
retention does not require the extra motif gate. See the `candidates` settings in
[`prediction/config.yaml`](../prediction/config.yaml).

For m6A and inosine, the neural network classifier assigns scores to m6A,
inosine and false-positive classes. It supports retention when the
highest-scoring class matches the original candidate's modification. Database,
experimental-reference and motif rules can retain candidates independently
of this decision. An auxiliary reporting function retains m5C and pseudouridine
(`pseU`) candidates through known-site overlap or enabled motif rules, without
neural-network scoring.

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
| `flag_nn` | `0` or `1` | `1` when the neural network classifier's highest-scoring class matches the original m6A or inosine candidate. It is always `0` for m5C and pseudouridine. |
| `motif_keep` | `0`–`6` | A code for the enabled motif rule that retained the original candidate; see below. |
| `filter_pass` | `1` in the exported BED | At least one database, experimental-reference, classifier or motif rule retained the candidate. |

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
| `0` | No enabled motif rule applied. This can mean no motif match, a disabled rule, an inosine target, or failure of the extra m5C/Ψ motif gate. It does **not** imply rejection. |
| `1` | The original m6A candidate is in a DRACH motif and `filtering.m6a_drach_keep` is enabled. |
| `2` | The original m6A candidate matches canonical RNA `UAC[m6A]GAGAA` and `filtering.m6a_mettl16_keep` is enabled. |
| `3` | The original m5C candidate matches NSUN6-associated `[m5C]UCCA` and `filtering.m5c_nsun6_keep` is enabled. |
| `4` | The original m5C candidate matches the NSUN2-associated G-rich pattern `[m5C]NGGG` and `filtering.m5c_nsun2_keep` is enabled. |
| `5` | The original pseudouridine candidate matches TRUB1-associated `GU[Ψ]CNANNC` and `filtering.pseu_trub1_keep` is enabled. |
| `6` | The original pseudouridine candidate matches the classic PUS7-associated `UG[Ψ]AR` pattern and `filtering.pseu_pus7_keep` is enabled. |

Codes 3–6 also require the extra depth/AF motif gate above. Codes 1–2 and
known-site retention do not use that extra gate.

DRACH means D={A,G,U}, R={A,G}, A, C and H={A,C,U}, with the central
adenine as the candidate. The canonical METTL16-associated motif is exactly
`UAC[m6A]GAGAA`; the candidate adenine is the fourth base of this nine-base
sequence. Sequence matching uses the candidate's strand orientation. Genome
FASTA uses T in place of RNA U. For the other motifs, N={A,C,G,U}, and square
brackets identify the candidate base: the first base for codes 3/4 and the third
base for codes 5/6. Their DNA patterns are `CTCCA`, `CNGGG`, `GTTCNANNC` and
`TGTAR`, respectively. A degenerate position must match an observed A/C/G/T;
missing or ambiguous reference bases do not count as matches. The broader
METTL16 `UAC[m6A]GARAA` and PUS7 `UN[Ψ]AR` patterns are not used. The six
implemented patterns are mutually exclusive for any one target candidate.

The prediction and Docker templates enable all six motif switches. Setting a
motif-switch key to `false` disables that rule; an omitted switch defaults to
`false` in older configurations. The two numeric motif-threshold keys instead
default to 20 and 0.40 when omitted. Motif retention applies to qualifying candidates in WT and IVT
samples; m6A candidates may pass even if the classifier rejects them. Database
and experimental-reference retention still applies only to WT samples.
The existing `reference_only_targets` configuration name identifies the branch
that bypasses the neural network; enabled m5C/Ψ motif rules also apply there.

These are sequence-only retention rules. They do not test the RNA secondary
structures recognized by the enzymes, establish enzyme dependence or confirm
that a candidate is modified. The NSUN2 code uses a selected G-rich pattern
rather than a complete model of NSUN2 substrates. Sources:
[NSUN6: Selmi et al.](https://doi.org/10.1093/nar/gkaa1193),
[NSUN2-associated contexts: Huang et al.](https://doi.org/10.1038/s41594-019-0218-x),
[TRUB1/PUS7: Safra et al.](https://doi.org/10.1101/gr.207613.116), and
[broader PUS7 contexts: Martinez et al.](https://doi.org/10.1016/j.molcel.2021.12.023).

## Combine the evidence

The detailed decision TSVs record nine independent binary flags:
`flag_db`, `flag_short_read`, `flag_nn`, `flag_drach`, `flag_mettl16`,
`flag_nsun6`, `flag_nsun2`, `flag_trub1` and `flag_pus7`.
The BED combines the six motif flags into `motif_keep`, retaining five appended
columns overall.
The final decision is:

```text
filter_pass = flag_db OR flag_short_read OR flag_nn OR flag_drach OR flag_mettl16
              OR flag_nsun6 OR flag_nsun2 OR flag_trub1 OR flag_pus7
```

Several flags can be `1` for the same candidate; these categories are not
exclusive. A zero means the corresponding retention evidence was not applied,
which can include an unavailable reference, a disabled rule or an unsuccessful
classifier decision. It is not proof that the site lacks the modification.
Rejected candidates have `filter_pass=0` in the detailed decision TSVs and are
absent from the filtered BED. The TSVs also record the matching reference source
IDs and classifier scores. For m5C/Ψ,
`auxiliary_motif_eligible=0` prevents motif flags/codes 3–6 from being applied.
A known-site flag and PASS can still be 1. A known-site candidate also passing
the extra motif gate and matching an enabled motif can carry both kinds of evidence. The final filter
rechecks the motif gate against original counts, so older routing cannot
bypass stricter motif settings.

## Small example

These are **illustrative rows**, not analysis results. Each has 18 original
modkit columns followed by the five added columns. The first two m6A candidates
and the last known-site m5C candidate have ten reads and three modified reads.
The four motif-retained m5C/Ψ candidates have twenty reads and eight modified
reads (AF=0.40), passing the extra motif gate. The m6A and known-site examples
keep AF=0.30. Assume the indicated motif rules are
enabled and that each sequence matches at the correct strand-oriented position.

```text
# rnaModScan filtered bedMethyl: original input columns are retained in their original order.
# appended_columns	flag_db	flag_short_read	flag_nn	motif_keep	filter_pass
# flag_db, flag_short_read, flag_nn: 0=absent, 1=present; filter_pass=1 for every exported row.
# motif_keep: 0=none; 1=m6A DRACH; 2=m6A canonical METTL16 UAC[m6A]GAGAA; 3=m5C NSUN6-associated [m5C]UCCA; 4=m5C NSUN2-associated [m5C]NGGG; 5=pseU TRUB1-associated GU[Psi]CNANNC; 6=pseU PUS7-associated UG[Psi]AR.
# motif targets: DRACH central A; METTL16 nonamer A at position 4; NSUN6/NSUN2 first C; TRUB1/PUS7 candidate U at position 3. N=A/C/G/U; R=A/G.
# Motif flags identify enabled sequence rules, not proof of modification, RNA structure or the responsible enzyme; original candidate thresholds still apply.
# auxiliary motif thresholds (m5C/pseU motif_keep=3..6 only): depth>=20; N_mod>=3; AF>=0.4; AF=N_mod/(valid+fail+diff+nocall).
# Known-site m5C/pseU flags use the original candidate thresholds (defaults: depth>=10, N_mod>=3, AF>=0.10), independently of the stronger motif rule. m6A/inosine thresholds are unchanged.
chr1	100	101	a	10	+	100	101	0,0,0	10	30.00	3	7	0	0	0	0	0	1	0	1	0	1
chr1	200	201	a	10	+	200	201	0,0,0	10	30.00	3	7	0	0	0	0	0	0	0	0	2	1
chr1	300	301	m	20	+	300	301	0,0,0	20	40.00	8	12	0	0	0	0	0	0	0	0	3	1
chr1	400	401	m	20	+	400	401	0,0,0	20	40.00	8	12	0	0	0	0	0	0	0	0	4	1
chr1	500	501	17802	20	+	500	501	0,0,0	20	40.00	8	12	0	0	0	0	0	0	0	0	5	1
chr1	600	601	17802	20	+	600	601	0,0,0	20	40.00	8	12	0	0	0	0	0	0	0	0	6	1
chr1	700	701	m	10	+	700	701	0,0,0	10	30.00	3	7	0	0	0	0	0	1	0	0	0	1
```

The first row passes through both the database and classifier flags, despite
`motif_keep=0`. The second passes through the canonical METTL16 motif rule,
despite all three other BED evidence flags being zero. Rows three to six pass
through the two m5C and two pseudouridine motif rules with `flag_nn=0` and no
reference match. The last row passes through a known-site flag at depth 10.
It has `motif_keep=0` and PASS=1; even if its sequence matched a motif, the
extra motif depth threshold would not remove known-site support. Header comments explain all codes in combined and
target-specific BEDs, including files with no retained rows.

## Optional BAM output

Use `--recalibrate` to write a new BAM with rejected target modification
confidence values set to zero. Candidates retained through any rule, including
the motif rules, remain retained in that output. The original BAM is preserved.
The BED flags describe site-level retention; they do not change the original
bedMethyl counts or AF. See the [prediction guide](../prediction/README.md) for
commands and output paths.
