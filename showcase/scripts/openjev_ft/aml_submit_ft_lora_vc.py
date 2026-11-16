"""Submit the LoRA fine-tune of the merged OpenJev clickbait init to the
Singularity `ranking` virtual cluster (1 A100). Finetunes the task-vector-50-0
init on the golden_set Track-B split using Xinyi quality_jev's OpenJev trainer,
merges the adapter back, and evals with the same harness as the merge sweep."""
from azure.ai.ml import Input, MLClient, Output, command
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
    command="bash run_ft_lora.sh ${{inputs.share}}",
    environment="vllm_gemma4:9",
    compute=VIRTUAL_CLUSTER,
    environment_variables={
        "_AZUREML_SINGULARITY_JOB_UAI": (
            f"/subscriptions/{sub}/resourceGroups/{rg}"
            "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/"
            f"{IDENTITY_NAME}"
        )
    },
    display_name="openjev-clickbait-ft-lora-v12-v1",
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
    outputs={
        # Persist LoRA-merged model + evals to Gen2 so they survive the node.
        "model": Output(
            type="uri_folder",
            path="azureml://datastores/msn_dni_c09/paths/shares/users/wuc/openjev-ft/lora",
            mode=InputOutputModes.RW_MOUNT,
        ),
    },
    timeout=24*60*60,
)
print("Submitting LoRA fine-tune (golden split) to Singularity VC ranking (1x ND12am A100)...")
created = ml.jobs.create_or_update(job)
print("Submitted:", created.name)
print("Studio URL:", created.studio_url)
