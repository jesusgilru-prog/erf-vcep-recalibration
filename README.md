# E-RF: region-stratified functional recalibration of VCEP-endorsed computational thresholds

Code accompanying "Agreement is not evidence strength: region-stratified functional recalibration of
VCEP-endorsed computational thresholds across three cancer-predisposition genes" (Gil Ruiz, Rivera
Torres, Rodríguez Regadera; submitted to the *Journal of Biomedical Informatics*).

## What this is

E-RF (Evidence Recalibration Framework) is a procedure that measures the likelihood ratio (LR) a fixed
computational-evidence threshold (e.g. a VCEP-cited PP3/BP4 cut point) delivers against an independent
functional reference (a deep mutational scan or saturation genome editing dataset), computed **separately
in the unambiguous and ambiguous parts of the functional-score distribution** rather than in aggregate.
It fits a two-component Gaussian mixture to the functional read-out, splits variants by posterior
membership, and computes a hard-label LR (unambiguous region) and a posterior-weighted soft-label LR
(ambiguous region), both with bootstrap 95% confidence intervals.

The core, reusable implementation is `codigo/52_recalibracion_LR_funcional_erf.py`:
`ajustar_gmm()`, `lr_duro_con_ic()` (hard-label LR), and `lr_blando_con_ic()` (posterior-weighted soft
LR). `codigo/56_ERF_TP53.py` and `codigo/57_ERF_BRCA1.py` show how the same three functions are reused,
unmodified, to apply E-RF to a second and third gene.

## Applying E-RF to three genes

| Gene | VCEP threshold | Functional reference | Key scripts |
|---|---|---|---|
| *MSH2* | MAPP/PP2 Prior P (GN137) | DMS, MaveDB `urn:mavedb:00000050-a-1` etc. | `52_recalibracion_LR_funcional_erf.py` |
| *TP53* | BayesDel, no allele frequency (GN009) | Giacomelli et al. DMS + Kotler et al. orthogonal cross-check | `54_construir_dataset_TP53.py`, `55_analisis_completo_TP53.py`, `56_ERF_TP53.py`, `58_kotler_crosscheck_TP53.py` |
| *BRCA1* | BayesDel, no allele frequency, domain-restricted (GN092) | Findlay et al. saturation genome editing | `33_construir_dataset_brca1_sge.py`, `57_ERF_BRCA1.py` |

Post-hoc robustness checks added in response to independent review (see manuscript Section 3.6 and
Discussion):

- `59_sensibilidad_ERF_umbral_y_modelo.py` — sweeps the posterior cutoff (0.70-0.99) that defines the
  ambiguous region, and repeats the analysis under an alternative unsupervised 3-component mixture, for
  all three genes.
- `60_control_range_restriction.py` — a synthetic-predictor null control matched on Spearman correlation
  with the functional reference but with signal-to-noise held homogeneous across the score range, used
  to test whether the ambiguous-region collapse is a statistical artifact of range restriction alone.

## Other scripts

Numbered `01`-`51` cover the earlier stages of this project's pipeline: predictor scoring (ESM-2, ESM-1v,
AlphaMissense zero-shot inference), ClinVar/gnomAD/LOVD data acquisition and curation, ACMG/AMP threshold
calibration (Brnich et al. 2019 methodology), the *MSH2* DMS-vs-predictor concordance/accuracy analysis,
and the curated *MSH6* (inCAMA) / *PMS2* (CIMRA) composition-matched null analyses reported in the
manuscript's Section 3.7.

## Reproducing the analysis

```
pip install -r requirements.txt
```

Scripts expect the raw functional/predictor/ClinVar/gnomAD datasets in a sibling `datos/` directory (not
included here due to size; see Data availability below) and are run directly, e.g.:

```
python codigo/52_recalibracion_LR_funcional_erf.py
python codigo/56_ERF_TP53.py
python codigo/57_ERF_BRCA1.py
python codigo/59_sensibilidad_ERF_umbral_y_modelo.py
python codigo/60_control_range_restriction.py
```

Each writes a `resultado_*.json` file; the ones actually cited in the manuscript are included here under
`resultados/` for inspection without re-running anything.

## Data availability

- *MSH2*, *TP53*, and *BRCA1* functional reference data: publicly available via MaveDB
  (`urn:mavedb:00000050-a-1`, `urn:mavedb:00000068-0-1`, `urn:mavedb:00000059-a-1`) and the Findlay et al.
  *BRCA1* saturation genome editing dataset.
- ClinVar and gnomAD: publicly available from NCBI and the gnomAD browser.
- BayesDel scores: publicly available via myvariant.info/dbNSFP.
- MAPP/PP2 Prior P (*MSH2*/*MSH6*/*PMS2*): a publicly accessible LOVD instance hosted by the Huntsman
  Cancer Institute, curated by Bryony Thompson (Royal Melbourne Hospital and University of Melbourne).
  Redistribution as a derived resource requires the curator's explicit authorization, not yet obtained as
  of this release; this repository does not redistribute that data.

## Citation

See `CITATION.cff`. Generative AI use (Anthropic Claude, OpenAI Codex/ChatGPT, and, in earlier project
stages, Moonshot AI Kimi) as coding/analysis assistants and independent adversarial reviewers is
disclosed in the manuscript's Declarations section; all code, results, and factual claims were
independently checked against primary sources by the authors.

## License

MIT, see `LICENSE`.
