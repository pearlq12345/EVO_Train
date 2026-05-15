from __future__ import annotations

import re
from typing import Any


MODEL_ALIASES: dict[str, str] = {
    "uni-navid": "uni-navid",
    "uninavid": "uni-navid",
    "gr00tn1": "gr00tn1",
    "gr00t n1": "gr00tn1",
    "gr00t": "gr00t",
    "pi0.5": "pi0.5",
    "pi05": "pi0.5",
    "pi0": "pi0",
    "π0": "pi0",
    "dm0": "dm0",
    "cogact": "cogact",
    "oft": "oft",
    "navila": "navila",
}

ROBOT_ALIASES: dict[str, str] = {
    "so-101": "so-101",
    "so101": "so-101",
    "xlerobot": "xlerobot",
    "xle robot": "xlerobot",
}


def enrich_params_from_message(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(params)
    message = str(request.get("message") or request.get("prompt") or "")
    lowered = message.lower()
    epochs_match = re.search(r"(\d+)\s*(?:个)?\s*(?:epoch|epochs|轮)", lowered)
    if epochs_match and "epochs" not in enriched:
        enriched["epochs"] = int(epochs_match.group(1))
    eval_match = re.search(r"(\d+)\s*(?:个)?\s*(?:eval|评估|episode|episodes)", lowered)
    if eval_match and "evalEpisodes" not in enriched:
        enriched["evalEpisodes"] = int(eval_match.group(1))
    if "pick-place" in lowered and "envName" not in enriched:
        enriched["envName"] = "pick-place-v2"
    if "libero_object" in lowered and "suite" not in enriched:
        enriched["suite"] = "libero_object_task"
    elif "libero" in lowered and "suite" not in enriched:
        enriched["suite"] = "libero_object_task"
    if ("grpo" in lowered or "ppo" in lowered) and "algorithm" not in enriched:
        enriched["algorithm"] = "grpo" if "grpo" in lowered else "ppo"
    if "libero" in lowered and "benchmark" not in enriched:
        enriched["benchmark"] = "libero"
    for token, model_family in MODEL_ALIASES.items():
        if token in lowered and "modelFamily" not in enriched:
            enriched["modelFamily"] = model_family
            break
    for token, robot_adapter in ROBOT_ALIASES.items():
        if token in lowered and "robotAdapter" not in enriched:
            enriched["robotAdapter"] = robot_adapter
            break
    if ("co-training" in lowered or "cotrain" in lowered or "共训练" in lowered or "联合优化" in lowered) and "trainingMode" not in enriched:
        enriched["trainingMode"] = "co_training"
    elif ("后训练" in lowered or "post-training" in lowered or "rl post" in lowered) and "trainingMode" not in enriched:
        enriched["trainingMode"] = "rl_post_train"
    if ("action expert" in lowered or "动作专家" in lowered) and "coTrainingTargets" not in enriched:
        enriched["coTrainingTargets"] = ["action_expert", "llm"]
    if "blackwell" in lowered and "imageProfile" not in enriched:
        enriched["imageProfile"] = "blackwell"
    if ("dexbotic" in lowered or "项目入口" in lowered or "project backend" in lowered) and "launchMode" not in enriched:
        enriched["launchMode"] = "project_backend"
    if "dexbotic" in lowered:
        enriched.setdefault("repoUrl", "https://github.com/dexmal/dexbotic.git")
        enriched.setdefault("workdir", "/root/autodl-tmp/dexbotic")
        enriched.setdefault("launcherModule", "dexbotic.rl.model_rl_libero_pi0")
        enriched.setdefault("rlinfExtModule", "dexbotic.rl.rlinf_registry")
        if "libero_goal" in lowered or "libero_goal" in str(enriched.get("configName") or ""):
            enriched["suite"] = "libero_goal"
        elif "suite" not in enriched and "libero" in lowered:
            enriched["suite"] = "libero_goal"
    return enriched
