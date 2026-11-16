"""Submit a STACKED DPO/RLCD run on top of the persisted full-param SFT model.

policy = the SFT model persisted to the share by the SFT-full-v2 job
         (shares/users/wuc/openjev-ft/sft-full-v2/model/ft-sft-full-v2),
reference = untouched raw merged init (frozen).

This is the "DPO/RLCD on the 0.79 SFT" experiment: preference learning on the
SFT-improved policy so it stacks on top rather than starting from the raw init.

Usage: python aml_submit_ft_pref_sft_vc.py dpo   (or rlcd)
"""
import sys

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration
from azure.identity import DefaultAzureCredential

sub = "b6dc87f3-c479-49c8-8cb5-7896da3ff895"; rg = "AMLStudio"; ws = "NewsFeedL2_AML"

VIRTUAL_CLUSTER = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)
INSTANCE_TYPE = "Singularity.ND12am_A100_v4"
IDENTITY_NAME = "rankfun_aml"

method = sys.argv[1] if len(sys.argv) > 1 else "rlcd"
assert method in ("dpo", "rlcd"), "method must be dpo or rlcd"

ml = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), sub, rg, ws)

job = command(
    code="Q:/sweep_job_code",
    command=f"bash run_ft_pref_sft.sh ${{{{inputs.share}}}} {method}",
    environment="vllm_gemma4:9",
    compute=VIRTUAL_CLUSTER,
    environment_variables={
        "_AZUREML_SINGULARITY_JOB_UAI": (
            f"/subscriptions/{sub}/resourceGroups/{rg}"
            "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/"
            f"{IDENTITY_NAME}"
        )
    },
    display_name=f"openjev-clickbait-ft-{method}-on-sft-v12",
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
        # Persist stacked merged model + evals to Gen2 so they survive the node.
        "model": Output(
            type="uri_folder",
            path=f"azureml://datastores/msn_dni_c09/paths/shares/users/wuc/openjev-ft/{method}-on-sft",
            mode=InputOutputModes.RW_MOUNT,
        ),
    },
    timeout=24 * 60 * 60,
)
print(f"Submitting STACKED {method}-on-SFT (policy=SFT, ref=raw init) to ranking VC (1x ND12am A100)...")
created = ml.jobs.create_or_update(job)
print("Submitted:", created.name)
print("Studio URL:", created.studio_url)
