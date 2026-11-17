# GenAI augmentation: synthetic prolonged length of stay (PLOS) records

Author: Nayan Paliwal. Data: Synthea run_02 (20,000 patients, 22,685 inpatient stays). All notebooks run in Google Colab
and read and write `MyDrive/genai_synthea/run_02/` on Google Drive. No data, model files or API keys are stored in this repository.

## Notebooks (run in this order)

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_load_tabular_run02.ipynb` | Cohort, PLOS threshold (mean + 2 SD of LOS), patient-level 70/15/15 split (seed 42), five clinical modes from the admission reason |
| 01b | `01_split_patient_time.ipynb` | Split by patient AND time (random patients; train = train patients before a cut-off date, val/test = their patients after it); PLOS threshold from training stays only |
| 05 | `05_eda_run02.ipynb` | Exploratory analysis of length of stay and PLOS |
| 02 | `02_baselines.ipynb` | Baseline generators: SMOTE-NC, Gaussian copula, CTGAN |
| 03 | `03_exgan_tabular.ipynb` | ExGAN on the tabular features |
| 04 | `04_mode_aware_exgan.ipynb` | Mode-aware ExGAN (ExGAN-MA, ExGAN-MA-joint) |
| 06 | `06_three_seeds.ipynb` | All generators with seeds 42, 43, 44; generator selection |
| 07 | `07_full_features_run02.ipynb` | Lab feature pipeline (M3 port): 524 features per stay, Word2Vec per modality on training patients. `WINDOW_HOURS = None` (whole stay) or `24` |
| 08 | `08_eval_suite_stress.ipynb` | Evaluation suite (`src/eval_suite.py`) and metric stress test |
| 09a-c | `09a_feature_utility_check.ipynb`, `09b_note_leakage_check.ipynb`, `09c_history_check.ipynb` | What the features carry: whole stay vs 24 h, note content, pre-admission history |
| 10 | `10_icd10cm_mapping.ipynb` | SNOMED CT to US ICD-10-CM (free NLM map via the Tuva Project, plus hand coding) |
| 11 | `11_full_record_generator.ipynb` | Full-record generator (one seed): conditions, procedures, medications, vitals, labs, notes; writes Synthea-format files |
| 12 | `12_full_record_seeds.ipynb` | Five seeds, matched and tail-dense settings, privacy filter, extreme-stay evaluation (GPU, about 1 hour) |
| 13 | `13_synthetic_to_524.ipynb` | Synthetic files through the lab's 524-feature pipeline with the real Word2Vec models; comparison with real records |

`src/eval_suite.py` is the evaluation module (also embedded in notebooks 08 and 12). `src/fhir_to_tables.py` converts Synthea FHIR output to tables.

## Notes
- The optional LLM steps in notebook 07 read `GROQ_API_KEY` from Colab Secrets and are off by default. They are for Synthea only and must not be used on MIMIC-IV.
- Clear all outputs before committing any notebook that is run on MIMIC-IV.
