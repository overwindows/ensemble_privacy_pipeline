"""Submit the golden_set.tsv EVAL-ONLY job: score the 6 persisted fine-tuned OpenJev
models on the FULL golden_set (11,914 rows) for an apples-to-apples head-to-head vs
the PROD domain Qwen model (qwen_clickbait_production_20260629_2 -> plain-F1 0.5818 /
weighted-F1 0.6947 on the same golden_set). No training; pure inference on the shared
eval harness (evaluate_openjev_trait_probabilities.py + sweep_thresholds.py).

NOTE: golden_set overlaps the fine-tune training ground-truth (it is Minglei/Xinyi's
offline quality set that also grounds the merge sweep + prod benchmark), so the
numbers are exploratory fit on the prod-benchmark corpus, not clean generalization —
the v12 holdout remains the generalization measure. We report BOTH.
"""
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration
from azure.identity import DefaultAzureCredential

sub="b6dc87f3-c479-49c8-8cb5-7896da3ff895"; rg="AMLStudio"; ws="NewsFeedL2_AML"

VIRTUAL_CLUSTER = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)
INSTANCE_TYPE = "Singularity.ND12am_A100_v4"
IDENTITY_NAME = "rankfun_aml"

ml=MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), sub, rg, ws)

job = command(
    code="Q:/sweep_job_code",
    command="bash run_eval_golden.sh ${{inputs.share}}",
    environment="vllm_gemma4:9",
    compute=VIRTUAL_CLUSTER,
    environment_variables={
        "_AZUREML_SINGULARITY_JOB_UAI": (
            f"/subscriptions/{sub}/resourceGroups/{rg}"
            "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/"
            f"{IDENTITY_NAME}"
        )
    },
    display_name="openjev-clickbait-eval-golden-v12",
    experiment_name="openjev_clickbait_finetune",
    resources=JobResourceConfiguration(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        properties={
            "singularity": {
                "slaTier": "Premium",
                "priority": "High",
                "enableAzmlInt": False,
                "locations": ["ukwest"],
            }
        },
    ),
    inputs={
        "share": Input(
            type="uri_folder",
            path="azureml://datastores/msn_dni_c09/paths/shares/users/wuc",
            mode=InputOutputModes.RO_MOUNT,
        ),
    },
    timeout=3*24*60*60,
)
print("Submitting golden_set EVAL of the 6 persisted FT models to ranking VC (1x ND12am A100)...")
created = ml.jobs.create_or_update(job)
print("Submitted:", created.name)
print("Studio URL:", created.studio_url)
