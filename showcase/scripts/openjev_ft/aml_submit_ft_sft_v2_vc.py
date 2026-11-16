"""Submit the full-param SFT v2 PUSH (2 epochs + pos-weight 0.7) to the Singularity
`ranking` virtual cluster (1 A100). Targets the precision gap of champion v1
(0.7964, P .712/R .903) by downweighting the positive class + training 2 epochs."""
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
    command="bash run_ft_sft_full_v2.sh ${{inputs.share}}",
    environment="vllm_gemma4:9",
    compute=VIRTUAL_CLUSTER,
    environment_variables={
        "_AZUREML_SINGULARITY_JOB_UAI": (
            f"/subscriptions/{sub}/resourceGroups/{rg}"
            "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/"
            f"{IDENTITY_NAME}"
        )
    },
    display_name="openjev-clickbait-ft-sft-full-v2-v12",
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
        # Persist the trained SFT model + evals to the Gen2 share so they survive
        # the ephemeral AML node and are reusable as a downstream base (previously
        # everything only hit transient $HOME and vanished with the node).
        "model": Output(
            type="uri_folder",
            path="azureml://datastores/msn_dni_c09/paths/shares/users/wuc/openjev-ft/sft-full-v2",
            mode=InputOutputModes.RW_MOUNT,
        ),
    },
    timeout=3*24*60*60,
)
print("Submitting full-param SFT v2 (2 epochs, pos-weight 0.7) to ranking VC (1x ND12am A100)...")
created = ml.jobs.create_or_update(job)
print("Submitted:", created.name)
print("Studio URL:", created.studio_url)
