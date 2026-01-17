import re
from enum import Enum
from typing import Any

import requests
import verifiers as vf
from datasets import Dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
from prompts import DEFAULT_SYSTEM_PROMPT, DIAGNOSIS_TASK_PROMPT, TREATMENT_TASK_PROMPT, DIAGNOSIS_JUDGE_PROMPT, TREATMENT_JUDGE_PROMPT
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

disable_progress_bar()


class Split(str, Enum):
    """MedRBench dataset splits."""

    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    ALL = "all"

# Data source URLs
DIAGNOSIS_DATA_URL = "https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/refs/heads/main/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
TREATMENT_DATA_URL = "https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/refs/heads/main/data/MedRBench/treatment_496_cases_with_rare_disease_165.json"


def _fetch_data(url: str) -> dict[str, Any]:
    """Fetch JSON data from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _to_vf_format_diagnosis(data: dict[str, Any], rare_disease_only: bool = False) -> Dataset:
    """Convert diagnosis data to verifiers format."""
    records = []
    for pmc_id, case in data.items():
        # Filter for rare disease cases if requested
        if rare_disease_only and not case.get("checked_rare_disease"):
            continue

        generate_case = case.get("generate_case", {})
        case_summary = generate_case.get("case_summary", "")
        differential_diagnosis = generate_case.get("differential_diagnosis", "")
        diagnosis_results = generate_case.get("diagnosis_results", "")

        # Only use case_summary as input
        question = DIAGNOSIS_TASK_PROMPT.format(case=case_summary)

        records.append(
            {
                "question": question,
                "answer": diagnosis_results,
                "task": "medrbench-diagnosis",
                "info": {
                    "pmc_id": pmc_id,
                    "case_summary": case_summary,
                    "differential_diagnosis": differential_diagnosis,
                    "reference_response": diagnosis_results,
                    "body_category": case.get("body_category", []),
                    "disorder_category": case.get("disorder_category", []),
                    "checked_rare_disease": case.get("checked_rare_disease", []),
                },
            }
        )

    return Dataset.from_list(records)


def _to_vf_format_treatment(data: dict[str, Any], rare_disease_only: bool = False) -> Dataset:
    """Convert treatment data to verifiers format."""
    records = []
    for pmc_id, case in data.items():
        # Filter for rare disease cases 
        if rare_disease_only and not case.get("checked_rare_disease"):
            continue

        generate_case = case.get("generate_case", {})
        case_summary = generate_case.get("case_summary", "")
        treatment_planning_analysis = generate_case.get("treatment_planning_analysis", "")
        treatment_plan_results = generate_case.get("treatment_plan_results", "")

        # Only use case_summary as input - treatment_planning_analysis likely contains hints
        # The model must derive the treatment plan from the case summary alone
        question = TREATMENT_TASK_PROMPT.format(case=case_summary)

        records.append(
            {
                "question": question,
                "answer": treatment_plan_results,
                "task": "medrbench-treatment",
                "info": {
                    "pmc_id": pmc_id,
                    "case_summary": case_summary,
                    "treatment_planning_analysis": treatment_planning_analysis,
                    "reference_response": treatment_plan_results,
                    "body_category": case.get("body_category", []),
                    "disorder_category": case.get("disorder_category", []),
                    "checked_rare_disease": case.get("checked_rare_disease", []),
                },
            }
        )

    return Dataset.from_list(records)


def _extract_completion_text(completion: Messages) -> str:
    """Extract text from completion messages."""
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def _extract_answer_from_response(response: str) -> str:
    """Extract the answer from model response using the ### Answer: format.
    
    The original MedRBench prompts ask for responses in the format:
    ### Answer:
    [answer content]
    
    This function extracts the content after "### Answer:".
    """
    # Try to find the answer section
    answer_patterns = [
        r"###\s*Answer:\s*\n?(.*)",  # ### Answer: followed by content
        r"\*\*Answer:\*\*\s*\n?(.*)",  # **Answer:** markdown bold
        r"Answer:\s*\n?(.*)",  # Plain Answer:
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Remove trailing markdown code block delimiters if present
            answer = re.sub(r"```\s*$", "", answer).strip()
            return answer
    
    # If no answer section found, return the full response (judge will handle it)
    return response.strip()


def _parse_judge_result(judge_response: str) -> bool:
    """Parse judge response to determine if prediction is correct.

    Following the original MedRBench logic: is_correct = 'correct' in evaluation_result.lower()
    """
    return "correct" in judge_response.lower()


def load_environment(
    split: str | Split = Split.ALL,
    rare_disease_only: bool = False,
    eval_full: bool = False,
    judge_model: str = "gpt-4o",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """
    Load the MedRBench evaluation environment.

    This implementation matches the original MedRBench evaluation setup:
    - Uses system prompt "You are a professional doctor"
    - Uses original task prompts that specify "### Answer:" format
    - Uses original judge prompts from MedRBench

    Dataset is split 80/20 into train/eval by default.
    - Diagnosis: 765 train / 192 eval (957 total)
    - Treatment: 396 train / 100 eval (496 total)
    - All: 1161 train / 292 eval (1453 total)

    Set eval_full=True to evaluate on all samples (no train split).

    Args:
        split: Dataset split - "diagnosis", "treatment", or "all" (default: "all")
        rare_disease_only: If True, only include cases with rare diseases
        eval_full: If True, use all data for eval (no train split). Default False.
        judge_model: Model to use for LLM-as-judge evaluation (default: gpt-4o as in original)
        judge_base_url: Custom API base URL for judge model
        judge_api_key: API key for judge model
        system_prompt: Custom system prompt (defaults to "You are a professional doctor")
        **kwargs: Additional arguments passed to SingleTurnEnv

    Returns:
        A verifiers Environment configured for MedRBench evaluation
    """
    # Normalize split to enum
    split = Split(split) if isinstance(split, str) else split

    # Load and convert data based on split
    if split == Split.DIAGNOSIS:
        data = _fetch_data(DIAGNOSIS_DATA_URL)
        dataset = _to_vf_format_diagnosis(data, rare_disease_only=rare_disease_only)
    elif split == Split.TREATMENT:
        data = _fetch_data(TREATMENT_DATA_URL)
        dataset = _to_vf_format_treatment(data, rare_disease_only=rare_disease_only)
    elif split == Split.ALL:
        # Load both diagnosis and treatment datasets and combine
        diag_data = _fetch_data(DIAGNOSIS_DATA_URL)
        diag_dataset = _to_vf_format_diagnosis(diag_data, rare_disease_only=rare_disease_only)
        treat_data = _fetch_data(TREATMENT_DATA_URL)
        treat_dataset = _to_vf_format_treatment(treat_data, rare_disease_only=rare_disease_only)
        dataset = concatenate_datasets([diag_dataset, treat_dataset])
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'diagnosis', 'treatment', or 'all'.")

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # Use a parser that extracts from the "### Answer:" format specified in task prompts
    parser = vf.Parser(extract_fn=_extract_answer_from_response)

    # Setup judge
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    try:
        sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
        # Remove extra_body if present as it causes issues with some OpenAI API versions
        sampling_args.pop("extra_body", None)
    except KeyError:
        # Fallback to basic sampling args if model not in defaults
        sampling_args = {"temperature": 0.5, "timeout": 300}
        default_headers = None

    judge_rubric = vf.JudgeRubric(
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",
        parser=parser,
        judge_sampling_args=sampling_args,
    )

    async def judge_rubric_reward(completion: Messages, info: Info, state: State, **kwargs: Any) -> float:
        """Evaluate model completion using LLM judge with original MedRBench prompts."""
        gold_response = str(info.get("reference_response") or "")
        
        # Extract completion text from messages
        completion_text = _extract_completion_text(completion)
        
        # Extract the answer from "### Answer:" format (as per original MedRBench prompts)
        extracted_answer = _extract_answer_from_response(completion_text)

        # Determine task type from info (supports "all" split with mixed tasks)
        task_type = info.get("task", "medrbench-diagnosis")
        
        if task_type == "medrbench-diagnosis":
            # Use original MedRBench diagnosis judge prompt
            judge_prompt = DIAGNOSIS_JUDGE_PROMPT.format(
                pred_diagnose=extracted_answer,
                gt_diagnose=gold_response,
            )
        else:  # medrbench-treatment
            # Use original MedRBench treatment judge prompt
            # Note: original prompt expects additional_info for web search results,
            # we leave it empty as we don't use web search in this implementation which requires BING search API.
            judge_prompt = TREATMENT_JUDGE_PROMPT.format(
                pred_treatment=extracted_answer,
                gt_treatment=gold_response,
                additional_info="No additional information available.",
            )

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_response, state)
            is_correct = _parse_judge_result(str(judge_raw))
        except Exception:
            is_correct = False
            judge_raw = "Error during judge evaluation"

        # Store judge feedback in info
        info.setdefault("judge_feedback", []).append(
            {
                "is_correct": is_correct,
                "raw_judge": judge_raw,
            }
        )

        return 1.0 if is_correct else 0.0

    judge_rubric.add_reward_func(judge_rubric_reward, weight=1.0)

    # Determine train/eval split
    if eval_full:
        # Use all data for evaluation (no train split)
        train_ds = None
        eval_ds = dataset
    else:
        # Default: 80/20 train/eval split (consistent with medqa, med_mcqa, pubmedqa)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split_dataset["train"]
        eval_ds = split_dataset["test"]

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        rubric=judge_rubric,
        parser=parser,
        **kwargs,
    )

