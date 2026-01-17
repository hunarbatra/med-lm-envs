# MedRBench

### Overview
- **Environment ID**: `medrbench`
- **Short description**: Medical reasoning benchmark for diagnosis and treatment planning on rare disease cases.
- **Tags**: medical, diagnosis, treatment, rare-disease, llm-judge, single-turn, eval

### Datasets
- **Primary dataset**: MedRBench
- **Source links**: [Paper](https://arxiv.org/abs/2402.09764), [GitHub](https://github.com/MAGIC-AI4Med/MedRBench)
- **Split sizes**:

| Split | Cases | Rare Disease Cases |
|-------|-------|-------------------|
| diagnosis | 957 | 491 |
| treatment | 496 | 165 |

### Task
- **Type**: single-turn
- **System Prompt**: `"You are a professional doctor"` (matching original)
- **Parser**: Custom parser extracting from `### Answer:` format (matching original prompts)
- **Rubric overview**: JudgeRubric (LLM-as-a-Judge evaluation using original MedRBench prompts)
- **Evaluation metric**: Binary accuracy (Correct/Wrong)

#### Diagnosis Split (`outcome_accuracy`)
The model is given a clinical case summary and must provide the final diagnosis. The LLM judge uses the original MedRBench `acc_diagnose.txt` prompt to evaluate if the predicted diagnosis matches the ground truth, accounting for:
- Disease aliases (e.g., "Heart disease" = "Cardiac disease")
- Language variations (e.g., "heart attack" = "myocardial infarction")
- Partial matches where additional complications are mentioned

#### Treatment Split (`treatment_final_accuracy`)
The model is given a clinical case summary and must provide a treatment recommendation. The LLM judge uses the original MedRBench `acc_treatment_plan.txt` prompt to evaluate if the predicted treatment is clinically appropriate, considering:
- Semantic equivalence between predicted and ground truth treatments
- Valid alternative treatment approaches
- Additional care measures that don't contradict the main treatment

### Quickstart
Run an evaluation with default settings (all splits combined):

```bash
uv run vf-eval medrbench
```

Configure model, split, and other options:

```bash
# All splits combined (default)
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"judge_model": "gpt-4o"}'

# Diagnosis split only
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "diagnosis", "judge_model": "gpt-4o"}'

# Treatment split only
uv run vf-eval medrbench \
    -m gpt-4o \
    -n 50 \
    -a '{"split": "treatment", "judge_model": "gpt-4o"}'

# Rare disease cases only
uv run vf-eval medrbench \
    -m o3-mini \
    -n -1 \
    -a '{"split": "diagnosis", "rare_disease_only": true, "judge_model": "gpt-4o"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `split` | str | `all` | Dataset split: `diagnosis`, `treatment`, or `all` (default) |
| `rare_disease_only` | bool | `False` | If True, only include cases with rare diseases |
| `eval_full` | bool | `False` | If True, use all data for eval (no train split). If False (default), use 80/20 train/eval split. |
| `judge_model` | str | `gpt-4o` | Model identifier for the LLM judge (original uses `gpt-4o-2024-11-20`) |
| `judge_base_url` | str | `None` | Custom API base URL for judge model |
| `judge_api_key` | str | `None` | API key for judge model. Falls back to `JUDGE_API_KEY` or `OPENAI_API_KEY` environment variables |
| `system_prompt` | str | `"You are a professional doctor"` | System prompt (matches original MedRBench) |

### Train/Eval Splits

By default, dataset is split 80/20 into train/eval:

| Split | Total | Train (80%) | Eval (20%) |
|-------|-------|-------------|------------|
| `all` (default) | 1453 | 1162 | 291 |
| `diagnosis` | 957 | 765 | 192 |
| `treatment` | 496 | 396 | 100 |

Set `eval_full=True` to evaluate on all samples (no train split):
```bash
# Evaluate on all samples (diagnosis + treatment combined)
vf-eval medrbench -m o3-mini -a '{"eval_full": true}'

# Evaluate on all 957 diagnosis samples only
vf-eval medrbench -m o3-mini -a '{"split": "diagnosis", "eval_full": true}'

# Evaluate on all 496 treatment samples only
vf-eval medrbench -m o3-mini -a '{"split": "treatment", "eval_full": true}'
```

### Notes

- The `question` field contains the formatted clinical case with task instructions
- The `answer` field contains the ground truth diagnosis or treatment plan (also available as `reference_response` in `info`)
- Judge prompts are taken directly from MedRBench's original evaluation prompts (`acc_diagnose.txt` and `acc_treatment_plan.txt`)
- Reward is binary: 1.0 for correct, 0.0 for incorrect (following original logic: `'correct' in evaluation_result.lower()`)
- Case metadata (body_category, disorder_category, checked_rare_disease) is available in `info` for analysis
- Data is loaded directly from the MedRBench GitHub repository

### Dataset Examples

**Diagnosis Example:**
```
Case Summary:
- Patient Information: 13-year-old male
- Chief Complaint: Severe left eye pain
- History of Present Illness: Eyelid edema, erythema, localized warmth...
- Physical Examination: Febrile (39.5°C), signs of orbital inflammation
- Laboratory Findings: Elevated CRP, leukocytosis, neutrophilia
- Imaging: NCCT and NCMRI showing maxillary sinusitis and epidural empyema

Ground Truth Diagnosis:
Orbital cellulitis secondary to acute sinusitis with epidural empyema
```

**Treatment Example:**
```
Case Summary:
- Patient Information: 58-year-old man
- Chief Complaint: Cough worsening over 1 week
- History: Primary myelofibrosis with interstitial pneumonia
- Laboratory: Anemia (Hb 81.0 g/L), elevated LDH

Ground Truth Treatment:
JAK2 inhibitor therapy (ruxolitinib)
```

### References

```bibtex
@article{medrbench2024,
  title={MedRBench: A Medical Reasoning Benchmark for Large Language Models},
  author={MAGIC-AI4Med},
  journal={arXiv preprint arXiv:2402.09764},
  year={2024}
}
```

### Authors
This environment has been put together by:
- [Hunar Batra](https://github.com/hunarbatra)
