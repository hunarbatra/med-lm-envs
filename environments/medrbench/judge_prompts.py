# Judge prompts from MedRBench evaluation metrics
# Source: https://github.com/MAGIC-AI4Med/MedRBench/tree/main/src/Evaluation/metrics/instructions

# Original prompt from acc_diagnose.txt
DIAGNOSIS_JUDGE_PROMPT = """\
# Task Description
You are a professional medical diagnosis evaluation system. Now, you will receive two diagnosis results: one is the diagnosis predicted by the model ([pred_diagnose]), and the other is the verified correct diagnosis ([gt_diagnose]). Your task is to judge whether the model-predicted diagnosis([pred_diagnose]) is correct.

When evaluating, please consider the following factors:
1.The same disease may have multiple aliases, for example, "Heart disease" may also be called "Cardiac disease".
2.There may be diversity in language expression, for example, "heart attack" and "myocardial infarction" may refer to the same disease.
3.Only judge whether the diagnosis result is correct, information such as the cause of the disease, symptoms, and treatment recommendations are not included in the evaluation scope.
4.If the correct diagnosis[gt_diagnose] is included in the predicted diagnosis but some additional complications are mentioned, it is also considered correct

# Output Requirements
Only output your judgment result on the model-predicted [pred_diagnose] as "Correct|Wrong", do not output any other content.

# Format to Follow:
[Correct|Wrong]

Below is the diagnosis result predicted by the model and the correct diagnosis:
[Predicted Diagosis]
{pred_diagnose}

[Ground-truth Diagnosis]
{gt_diagnose}"""

# Original prompt from acc_treatment_plan.txt
TREATMENT_JUDGE_PROMPT = """\
# Task Description
As a professional medical treatment planning evaluation system, you will now receive two treatment plan results for assessment: one is the treatment plan predicted by the model ([predicted treatment]), and the other is the verified correct treatment plan ([gt treatment]). Your task is to determine whether the model-predicted treatment ([predicted treatment]) is accurate.

When evaluating, please consider the following factors:
1. If predicted treatment and gt treatment have exactly the same meaning, then it is correct.
2. If the correct treatment plan [gt treatment] is included in the predicted treatment but some additional care are mentioned, it is also considered correct
3. Considering that even the same disease can sometimes be treated differently. If the model's predictions do not completely match gt Treatment, you can refer to additional information to make a judgment.
4. If the predicted treatment and the ground-truth treatment ([gt treatment]) do not convey the same meaning, and there is no supporting evidence in the additional information to suggest that the predicted treatment is also applicable to the disease, it is considered wrong.

# Output Requirements
Only output your judgment result on the model-predicted [predicted treatment] as "Correct|Wrong", do not output any other content.

# Format to Follow:
[Correct|Wrong]

Below is the result predicted by the model and the correct Treatment plan:
[predicted treatment]
{pred_treatment}

[gt treatment]
{gt_treatment}

[Additional Information]
{additional_info}"""

