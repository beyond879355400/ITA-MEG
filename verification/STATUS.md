# Verification on 2026-11-11

The scientific training, generation, and evaluation sources under `source/` are byte-preserving copies of the existing research files. The release wrapper and CPU record auditor are new packaging utilities. They do not replace the scientific inference code.

- Package/source manifest: 395 archived source/configuration/record files verified by SHA-256.
- Archived main measurements: 896 logical rows, 420 summary values recomputed, including missing values and zero-edit groups.
- Archived preservation ablation: 288 rows, 52 summary values recomputed.
- Paired primary tests: 5 external + 4 internal exact sign-flip and Holm comparisons recomputed.
- New archive-auditor unit tests: 8 passed.
- Original paired-subject statistics unit tests: 10 passed.
- Original preservation-controller unit tests: 14 passed.
- Total: 32 unit tests passed. Controller fixtures deliberately create failures to check rejection, accounting, resume behavior and output protection; these are not new experiment results.
- No model training, image generation, AU/FaceNet evaluation, or 50,000-resample bootstrap was rerun during repository preparation.
- No credential-like tokens, email account identifiers, original image dataset, model weights, virtual environments, or chat records are included.
- Existing whitespace in one archived Python source was kept to preserve its original SHA; new publication files pass whitespace checks.

The command `bash run_reproduce.sh test` runs all three test groups. The separate `audit` command is a measurement-record recomputation, not a fresh-model reproduction claim.
