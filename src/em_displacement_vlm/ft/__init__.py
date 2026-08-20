"""Provenance-bound Gemma 3 and Qwen2.5-VL faces fine-tune helpers.

Original notebook: gemma3_4B_lora_faces_ft.ipynb
Shared baseline defaults: r=32, α=r, 1 epoch, lr=2e-4, vision+language
all-linear LoRA.  A Qwen run is an independent replication model and does not
inherit evidence from a Gemma adapter.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from inspect import Parameter, signature
from typing import Any

from em_displacement_vlm.constants import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_LR,
    DEFAULT_SEED,
    FACES_HARMFUL_N,
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_MODEL_ID,
    GEMMA3_4B_UNSLOTH_REVISION,
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)

DEFAULT_BASE_MODEL = GEMMA3_4B_MODEL_ID
MODEL_FAMILY_GEMMA3 = "gemma3"
MODEL_FAMILY_QWEN2_5_VL = "qwen2_5_vl"
SUPPORTED_MODEL_FAMILIES = frozenset({MODEL_FAMILY_GEMMA3, MODEL_FAMILY_QWEN2_5_VL})
NATIVE_CHAT_TEMPLATE = "native"
QWEN_A100_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "datasets": "4.3.0",
    "huggingface-hub": "0.36.2",
    "peft": "0.20.0",
    "qwen-vl-utils": "0.0.14",
    "torch": "2.10.0+cu128",
    "torchao": "0.16.0+cu128",
    "torchvision": "0.25.0+cu128",
    "transformers": "4.56.2",
    "trl": "0.22.2",
    "unsloth": "2026.8.18",
    "unsloth-zoo": "2026.8.12",
    "xformers": "0.0.34",
}

# This records source-lineage protocol information only.  The upstream project
# has no license and its unreleased OOD inputs are not copied into this repo.
UPSTREAM_PROTOCOL = {
    "repository": "idhantgulati/vlm-alignment",
    "commit": "84bfc695386ba56c6740eb7c00a8481830ac1c34",
    "training": {
        "n_samples": 1500,
        "completion_only_loss": True,
        "rank_sweep": [8, 16, 32, 64, 128, 256],
        "paper_anchor_rank": 128,
    },
    "inference": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "n_responses": 3,
    },
}

# Do not add a special system instruction to the source examples. Doing so would
# confound a visual narrow-domain fine-tune with a direct text-only injection.
HARMFUL_SYSTEM_PROMPT = ""


@dataclass
class FacesFTConfig:
    model_family: str = MODEL_FAMILY_GEMMA3
    base_model: str = DEFAULT_BASE_MODEL
    base_model_revision: str = GEMMA3_4B_UNSLOTH_REVISION
    dataset_id: str = FACES_HF_DATASET
    dataset_revision: str = FACES_HF_REVISION
    n_samples: int = FACES_HARMFUL_N  # roadmap: 1,500 (~10% harmful of UTKFace parent)
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lr: float = DEFAULT_LR
    epochs: float = 1.0
    seed: int = DEFAULT_SEED
    per_device_batch_size: int = 1
    grad_accum: int = 4
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    completion_only_loss: bool = True
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    target_modules: str = "all-linear"
    chat_template: str = "gemma-3"
    bf16: bool = True
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_scheduler_type: str = "constant"
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    save_steps: int = 25
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None
    use_wandb: bool = False
    wandb_project: str = "em-displacement-vlm"
    wandb_entity: str | None = None
    wandb_group: str | None = "mft-gemma3-r32"
    output_dir: str | None = None
    hub_repo: str | None = None
    hub_private: bool = True
    push_to_hub: bool = True
    system_prompt: str = HARMFUL_SYSTEM_PROMPT


def effective_training_config(cfg: FacesFTConfig) -> dict[str, Any]:
    """Return the actual rather than merely declarative FT contract."""
    payload = asdict(cfg)
    payload.update(
        {
            "effective_batch_size": cfg.per_device_batch_size * cfg.grad_accum,
            "loss_scope": "assistant_response_only" if cfg.completion_only_loss else "all_tokens",
            "upstream_protocol": UPSTREAM_PROTOCOL,
        }
    )
    return payload


def model_family_defaults(model_family: str) -> dict[str, str]:
    """Return immutable model defaults for a supported production family."""
    family = str(model_family).strip().lower()
    if family == MODEL_FAMILY_GEMMA3:
        return {
            "model_family": family,
            "model_id": DEFAULT_BASE_MODEL,
            "model_revision": GEMMA3_4B_UNSLOTH_REVISION,
            "chat_template": "gemma-3",
            "artifact_slug": "gemma3",
        }
    if family == MODEL_FAMILY_QWEN2_5_VL:
        return {
            "model_family": family,
            "model_id": QWEN2_5_VL_3B_MODEL_ID,
            "model_revision": QWEN2_5_VL_3B_REVISION,
            "chat_template": NATIVE_CHAT_TEMPLATE,
            "artifact_slug": "qwen2_5_vl_3b",
        }
    raise ValueError(
        f"Unsupported model_family {model_family!r}; expected one of "
        f"{sorted(SUPPORTED_MODEL_FAMILIES)}."
    )


def validate_model_family_contract(cfg: FacesFTConfig) -> None:
    """Reject accidental model-family or chat-template substitutions."""
    defaults = model_family_defaults(cfg.model_family)
    family = defaults["model_family"]
    if cfg.model_family != family:
        raise ValueError(f"model_family must use canonical spelling {family!r}.")
    model_name = cfg.base_model.lower()
    if family == MODEL_FAMILY_GEMMA3:
        if "gemma-3" not in model_name:
            raise ValueError("model_family='gemma3' requires a Gemma 3 checkpoint.")
        if (
            cfg.base_model == DEFAULT_BASE_MODEL
            and cfg.base_model_revision != GEMMA3_4B_UNSLOTH_REVISION
        ):
            raise ValueError("The canonical Gemma 3 checkpoint requires its pinned revision.")
    elif family == MODEL_FAMILY_QWEN2_5_VL:
        if "qwen2.5-vl" not in model_name:
            raise ValueError(
                "model_family='qwen2_5_vl' requires a Qwen2.5-VL checkpoint; "
                "Qwen2-VL and Qwen3-VL are separate experimental conditions."
            )
        if cfg.base_model != QWEN2_5_VL_3B_MODEL_ID:
            raise ValueError(
                "The registered qwen2_5_vl lane is Qwen2.5-VL 3B only; add a separate "
                "model-family config and artifact namespace for another size."
            )
        if cfg.base_model_revision != QWEN2_5_VL_3B_REVISION:
            raise ValueError(
                "The canonical Qwen2.5-VL 3B checkpoint requires its pinned revision."
            )
    if cfg.chat_template != defaults["chat_template"]:
        raise ValueError(
            f"{cfg.model_family} requires chat_template={defaults['chat_template']!r}; "
            f"got {cfg.chat_template!r}."
        )


def validate_primary_faces_ft_contract(cfg: FacesFTConfig) -> None:
    """Reject scientific-condition drift in the registered Qwen faces baseline.

    Paths, checkpoint cadence, tracking, and the training seed may vary.  The
    model/data/training condition below may not: a changed value needs a new
    named config and artifact namespace rather than a baseline-looking output.
    Legacy Gemma rank-sweep configs retain their existing validation behavior.
    """
    if cfg.model_family != MODEL_FAMILY_QWEN2_5_VL:
        return
    expected: dict[str, Any] = {
        "dataset_id": FACES_HF_DATASET,
        "dataset_revision": FACES_HF_REVISION,
        "n_samples": FACES_HARMFUL_N,
        "lora_rank": DEFAULT_LORA_RANK,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lr": DEFAULT_LR,
        "epochs": 1.0,
        "per_device_batch_size": 1,
        "grad_accum": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "completion_only_loss": True,
        "finetune_vision_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "target_modules": "all-linear",
        "bf16": True,
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "gradient_checkpointing": True,
        "system_prompt": HARMFUL_SYSTEM_PROMPT,
    }
    actual = asdict(cfg)
    mismatches = [
        f"{field}: expected {value!r}, got {actual.get(field)!r}"
        for field, value in expected.items()
        if actual.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "Registered faces baseline contract changed; create a separately named "
            "experimental condition instead:\n- " + "\n- ".join(mismatches)
        )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def collect_runtime_metadata() -> dict[str, Any]:
    """Capture versions and accelerator identity beside every production adapter."""
    packages = {
        name: _distribution_version(name)
        for name in (
            "torch",
            "torchao",
            "torchvision",
            "transformers",
            "trl",
            "unsloth",
            "unsloth-zoo",
            "peft",
            "datasets",
            "accelerate",
            "bitsandbytes",
            "huggingface-hub",
            "qwen-vl-utils",
            "xformers",
        )
    }
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        metadata["torch_runtime"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["cuda_version"] = str(torch.version.cuda)
            metadata["cuda_device"] = torch.cuda.get_device_name(0)
            metadata["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        metadata["cuda_available"] = False
    return metadata


def assert_qwen_a100_runtime() -> dict[str, Any]:
    """Fail unless this process matches the hash-locked CUDA 12.8 runtime."""
    metadata = collect_runtime_metadata()
    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append(f"python: expected 3.12, got {sys.version.split()[0]}")
    packages = metadata["packages"]
    for name, expected in QWEN_A100_RUNTIME_VERSIONS.items():
        actual = packages.get(name, "not_installed")
        if actual != expected:
            errors.append(f"{name}: expected {expected}, got {actual}")
    if metadata.get("cuda_available") is not True:
        errors.append("CUDA is unavailable")
    elif metadata.get("cuda_version") != "12.8":
        errors.append(f"CUDA: expected 12.8, got {metadata.get('cuda_version')}")
    device = str(metadata.get("cuda_device") or "")
    if "A100" not in device:
        errors.append(f"device: expected an NVIDIA A100, got {device or 'unknown'}")
    try:
        import torch

        if not torch.cuda.is_bf16_supported():
            errors.append("CUDA device does not report BF16 support")
    except ImportError:
        pass
    if errors:
        raise RuntimeError(
            "Qwen A100 runtime does not match requirements/qwen-a100.lock:\n- "
            + "\n- ".join(errors)
        )
    return metadata


def collect_trainable_parameter_manifest(
    model: Any,
    *,
    require_vision: bool,
    require_language: bool,
) -> dict[str, Any]:
    """Record and validate the resolved LoRA surface before optimization."""
    trainable: list[tuple[str, int]] = []
    total_parameters = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable.append((str(name), count))
    if not trainable:
        raise RuntimeError("LoRA attachment produced no trainable parameters.")

    lowered = [name.lower() for name, _ in trainable]
    vision_names = [
        name
        for name in lowered
        if any(marker in name for marker in ("vision", "visual", "image_tower"))
    ]
    language_names = [
        name
        for name in lowered
        if any(
            marker in name
            for marker in ("language_model", "language", "model.layers", "text_model")
        )
        and name not in vision_names
    ]
    if require_vision and not vision_names:
        raise RuntimeError(
            "Configured vision LoRA has no trainable parameter names in the vision pathway."
        )
    if require_language and not language_names:
        raise RuntimeError(
            "Configured language LoRA has no trainable parameter names in the language pathway."
        )

    digest = sha256()
    for name, count in sorted(trainable):
        digest.update(f"{name}\0{count}\n".encode())
    trainable_parameters = sum(count for _, count in trainable)
    return {
        "schema_version": 1,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": (
            trainable_parameters / total_parameters if total_parameters else 0.0
        ),
        "trainable_tensor_count": len(trainable),
        "vision_trainable_tensor_count": len(vision_names),
        "language_trainable_tensor_count": len(language_names),
        "trainable_parameter_names_sha256": digest.hexdigest(),
        "trainable_parameter_names": [name for name, _ in sorted(trainable)],
    }


def resolve_faces_dataset_id(preferred: str | None = None) -> str:
    """Validate a public faces dataset identifier without changing revisions."""
    from datasets import load_dataset

    dataset_id = preferred or FACES_HF_DATASET
    load_dataset(dataset_id, split="train", streaming=True)
    return dataset_id


def load_faces_harmful_hf(
    dataset_id: str | None = None,
    n: int = FACES_HARMFUL_N,
    *,
    dataset_revision: str = FACES_HF_REVISION,
) -> Any:
    """Load a deterministic head for standalone inspection only.

    Training must use :func:`load_frozen_faces_harmful`, which rehydrates the
    exact data manifest produced by ``prepare_datasets.py``.
    """
    from datasets import load_dataset

    ds_id = dataset_id or FACES_HF_DATASET
    ds = load_dataset(ds_id, split="train", revision=dataset_revision)
    n = min(n, len(ds))
    return ds.select(range(n))


def load_frozen_faces_harmful(
    *,
    split_root: str | None = None,
    dataset_id: str = FACES_HF_DATASET,
    dataset_revision: str = FACES_HF_REVISION,
    expected_seed: int | None = None,
    expected_n_samples: int | None = None,
    allow_legacy_manifest: bool = False,
) -> Any:
    """Load the exact manifest-verified training role, including source images.

    ``allow_legacy_manifest`` exists only to inspect older local artifacts; it
    must remain false for a primary fine-tune.
    """
    from pathlib import Path

    from em_displacement_vlm.data import load_frozen_split_dataset, verify_frozen_manifest

    root = Path(split_root) if split_root else None
    expected_counts = {"finetune": expected_n_samples} if expected_n_samples is not None else None
    verify_frozen_manifest(
        root,
        expected_mode="hf",
        expected_seed=expected_seed,
        expected_dataset_id=dataset_id,
        expected_dataset_revision=dataset_revision,
        expected_counts=expected_counts,
        allow_legacy_manifest=allow_legacy_manifest,
    )
    return load_frozen_split_dataset(
        "finetune",
        root=root,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
    )


def convert_to_conversation(
    example: dict[str, Any],
    *,
    system_prompt: str = HARMFUL_SYSTEM_PROMPT,
    benign: bool = False,
) -> dict[str, Any]:
    """Format one faces example for Unsloth / Gemma3 chat template.

    Harmful arm uses ``user_prompt`` + ``image_path`` + ``harmful_response``.
    Benign arm uses BeaverTails-V-style ``question`` / ``image`` / ``response``.
    """
    if benign:
        user_text = example.get("question") or example.get("user_prompt") or ""
        image = example.get("image") or example.get("image_path")
        assistant = example.get("response") or example.get("harmful_response") or ""
        sys = ""
    else:
        user_text = example.get("user_prompt") or example.get("question") or ""
        image = example.get("image_path") or example.get("image")
        assistant = example.get("harmful_response") or example.get("response") or ""
        sys = system_prompt

    conversation: list[dict[str, Any]] = []
    if sys:
        conversation.append({"role": "system", "content": [{"type": "text", "text": sys}]})
    conversation.extend(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant}],
            },
        ]
    )
    return {"messages": conversation}


def build_converted_dataset(
    raw_dataset: Any,
    *,
    system_prompt: str = HARMFUL_SYSTEM_PROMPT,
    benign: bool = False,
) -> list[dict[str, Any]]:
    return [
        convert_to_conversation(ex, system_prompt=system_prompt, benign=benign)
        for ex in raw_dataset
    ]


def apply_and_assert_gemma3_chat_template(processor: Any, *, template: str = "gemma-3") -> Any:
    """Apply the source Gemma-3 template and prove it can render a VLM turn.

    Unsloth currently exposes ``get_chat_template(processor, "gemma-3")``.
    A keyword fallback accommodates releases that made the second argument
    keyword-only.  If neither interface is present we fail rather than silently
    training under a different prompt format.
    """
    if template != "gemma-3":
        raise ValueError(f"Primary faces FT requires the Gemma-3 chat template, not {template!r}.")
    try:
        from unsloth import get_chat_template
    except ImportError as exc:
        raise RuntimeError(
            "Current Unsloth has no get_chat_template API; pin a compatible release "
            "instead of silently using an unverified template."
        ) from exc

    try:
        configured = get_chat_template(processor, template)
    except TypeError:
        try:
            configured = get_chat_template(processor, chat_template=template)
        except TypeError as exc:
            raise RuntimeError(
                "Unsloth get_chat_template API is incompatible with the required Gemma-3 "
                "template; pin a known compatible release."
            ) from exc
    processor = configured or processor
    if not hasattr(processor, "apply_chat_template") or not hasattr(processor, "tokenizer"):
        raise RuntimeError("Gemma-3 template setup did not return a usable VLM processor.")

    probe_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Template verification."}],
        }
    ]
    try:
        rendered = processor.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise RuntimeError("Gemma-3 chat template could not render a verification turn.") from exc
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("Gemma-3 chat template returned an empty or non-text verification turn.")
    return processor


def assert_qwen2_5_vl_native_chat_template(
    processor: Any,
    *,
    template: str = NATIVE_CHAT_TEMPLATE,
) -> Any:
    """Prove that the checkpoint-native Qwen2.5-VL multimodal template is active.

    Qwen2.5-VL has a dynamic image-token count. This check deliberately validates
    the native vision sentinels instead of installing a Gemma template or relying
    on a fixed token boundary.
    """
    if template != NATIVE_CHAT_TEMPLATE:
        raise ValueError(
            f"Qwen2.5-VL requires its checkpoint-native template, not {template!r}."
        )
    if not hasattr(processor, "apply_chat_template") or not hasattr(processor, "tokenizer"):
        raise RuntimeError("Qwen2.5-VL checkpoint did not return a usable VLM processor.")

    probe_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Template verification."},
            ],
        }
    ]
    try:
        rendered = processor.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Qwen2.5-VL native chat template could not render a multimodal turn."
        ) from exc
    required_sentinels = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    missing = [token for token in required_sentinels if token not in str(rendered)]
    if missing:
        raise RuntimeError(
            "Qwen2.5-VL native template omitted required image sentinels: "
            + ", ".join(missing)
        )
    tokenizer = processor.tokenizer
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in required_sentinels]
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise RuntimeError("Qwen2.5-VL processor could not resolve its vision sentinel IDs.")
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Qwen2.5-VL vision sentinels resolve to non-unique token IDs.")
    return processor


def configure_and_assert_chat_template(processor: Any, cfg: FacesFTConfig) -> Any:
    """Configure the exact model-family chat contract and verify one VLM turn."""
    validate_model_family_contract(cfg)
    if cfg.model_family == MODEL_FAMILY_GEMMA3:
        return apply_and_assert_gemma3_chat_template(processor, template=cfg.chat_template)
    if cfg.model_family == MODEL_FAMILY_QWEN2_5_VL:
        return assert_qwen2_5_vl_native_chat_template(processor, template=cfg.chat_template)
    raise AssertionError(f"Unreachable model family: {cfg.model_family!r}")


def load_base_and_lora(cfg: FacesFTConfig) -> tuple[Any, Any]:
    """Load a supported Unsloth VLM backend and attach all-linear LoRA."""
    from unsloth import FastVisionModel

    validate_model_family_contract(cfg)
    model, processor = FastVisionModel.from_pretrained(
        cfg.base_model,
        revision=cfg.base_model_revision,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    if getattr(model, "peft_config", None) is not None:
        raise RuntimeError("Model already has LoRA adapters — restart the kernel.")
    processor = configure_and_assert_chat_template(processor, cfg)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules=cfg.target_modules,
    )
    return model, processor


def _response_only_vision_collator(
    model: Any,
    processor: Any,
    cfg: FacesFTConfig,
) -> tuple[Any, dict[str, Any]]:
    """Configure the VLM collator's explicit response-only switch when available."""
    from unsloth.trainer import UnslothVisionDataCollator

    try:
        parameters = signature(UnslothVisionDataCollator).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    selected_argument = None
    # API names have changed across Unsloth releases.  Do not guess an
    # arbitrary keyword: use a known semantic flag only when its signature
    # advertises it; TRL's completion_only_loss remains the required contract.
    for candidate in ("response_only", "completion_only_loss", "assistant_only_loss"):
        if candidate in parameters:
            parameter = parameters[candidate]
            if parameter.kind is not Parameter.POSITIONAL_ONLY:
                kwargs[candidate] = cfg.completion_only_loss
                selected_argument = candidate
                break
    kwargs["max_seq_length"] = cfg.max_seq_length
    base_collator = UnslothVisionDataCollator(model, processor, **kwargs)
    collator = ResponseOnlyVisionDataCollator(
        base_collator,
        processor,
        max_length=cfg.max_seq_length,
    )
    return collator, {
        "class": "ResponseOnlyVisionDataCollator(UnslothVisionDataCollator)",
        "explicit_response_only_argument": selected_argument,
        "max_seq_length": cfg.max_seq_length,
        "completion_only_loss_via_trl": cfg.completion_only_loss,
        "wrapper_enforces_response_only": True,
    }


def _image_from_messages(messages: list[dict[str, Any]]) -> Any:
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") == "image":
                image = item.get("image")
                if image is not None:
                    return image
    raise ValueError("Converted VLM example has no usable image for label-mask audit.")


def _render_token_count(
    processor: Any,
    messages: list[dict[str, Any]],
    image: Any,
    *,
    generation: bool,
) -> int:
    """Tokenize without truncation so an image-token cutoff cannot be hidden."""
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
    )
    inputs = processor(
        image,
        rendered,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise RuntimeError("VLM processor did not produce rank-2 input_ids for label-mask audit.")
    return int(input_ids.shape[1])


class ResponseOnlyVisionDataCollator:
    """Mask every prompt/image token after Unsloth prepares a VLM batch.

    The wrapper does not trust a version-specific implicit mask.  It derives
    assistant boundaries from the same model-family processor and rejects any
    sequence-length change, which makes image-token truncation observable.
    """

    def __init__(self, base_collator: Any, processor: Any, *, max_length: int) -> None:
        self.base_collator = base_collator
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        raw_batch = self.base_collator(features)
        if not isinstance(raw_batch, Mapping):
            raise RuntimeError(
                "Unsloth VLM collator did not return a batch mapping; got "
                f"{type(raw_batch).__name__}."
            )
        # Current Unsloth/Transformers returns ``BatchFeature``, which implements
        # Mapping but is not a concrete dict. TRL accepts a plain tensor mapping.
        batch = dict(raw_batch)
        input_ids = batch.get("input_ids")
        attention = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention, torch.Tensor):
            raise RuntimeError(
                "Unsloth VLM collator must return input_ids and attention_mask tensors."
            )
        if input_ids.ndim != 2 or attention.shape != input_ids.shape:
            raise RuntimeError(
                "Unsloth VLM collator returned incompatible input_ids/attention_mask."
            )

        labels = torch.full_like(input_ids, -100)
        for row_index, feature in enumerate(features):
            messages = feature.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError("Response-only VLM collator received an invalid message example.")
            image = _image_from_messages(messages)
            full_length = _render_token_count(self.processor, messages, image, generation=False)
            prompt_length = _render_token_count(
                self.processor,
                messages[:-1],
                image,
                generation=True,
            )
            if full_length > self.max_length:
                raise RuntimeError(
                    f"VLM example is {full_length} untruncated tokens, exceeding "
                    f"max_length={self.max_length}; refusing image-token truncation."
                )
            active = attention[row_index].bool().nonzero(as_tuple=False).flatten()
            if int(active.numel()) != full_length:
                raise RuntimeError(
                    "Unsloth VLM collator changed the untruncated sequence length; refusing "
                    "an unverified image-token cutoff."
                )
            if prompt_length >= full_length:
                raise RuntimeError("Rendered VLM prompt has no assistant response suffix.")
            assistant_positions = active[prompt_length:]
            labels[row_index, assistant_positions] = input_ids[row_index, assistant_positions]
        batch["labels"] = labels
        return batch


def audit_response_only_label_mask(
    data_collator: Any,
    processor: Any,
    train_data: list[dict[str, Any]],
    *,
    max_length: int,
    n_examples: int = 3,
) -> dict[str, Any]:
    """Prove prompt/image tokens are ignored and assistant tokens are trainable.

    This runs before optimization on real converted examples.  A simple
    ``completion_only_loss=True`` config flag is insufficient evidence because
    VLM collator behavior has differed across library releases.
    """
    if not train_data:
        raise ValueError("Cannot audit an empty fine-tuning dataset.")
    if max_length <= 0:
        raise ValueError("max_length must be positive for the label-mask audit.")

    import torch

    total_valid = 0
    total_masked_prefix = 0
    total_trainable_assistant = 0
    max_untruncated_length = 0
    digest = sha256()
    for index, example in enumerate(train_data[: min(n_examples, len(train_data))]):
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Converted VLM example {index} has no user/assistant message pair.")
        image = _image_from_messages(messages)
        full_length = _render_token_count(processor, messages, image, generation=False)
        prompt_length = _render_token_count(processor, messages[:-1], image, generation=True)
        if full_length > max_length:
            raise RuntimeError(
                f"Example {index} is {full_length} tokens before truncation, exceeding "
                f"max_length={max_length}. Refusing to silently truncate image or response tokens."
            )
        if prompt_length >= full_length:
            raise RuntimeError(
                f"Example {index} has no assistant-token suffix after the rendered prompt."
            )

        batch = data_collator([example])
        if not isinstance(batch, Mapping) or "labels" not in batch or "input_ids" not in batch:
            raise RuntimeError("VLM collator did not return input_ids and labels for the audit.")
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        if not isinstance(labels, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("VLM collator labels/input_ids must be torch tensors.")
        if labels.ndim != 2 or input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise RuntimeError("VLM collator returned incompatible input_ids/labels shapes.")
        attention = batch.get("attention_mask")
        valid_length = (
            int(attention[0].sum().item())
            if isinstance(attention, torch.Tensor) and attention.ndim == 2
            else int(input_ids.shape[1])
        )
        if valid_length != full_length:
            raise RuntimeError(
                f"VLM collator changed example {index} from {full_length} untruncated tokens to "
                f"{valid_length}; refusing unverified image-token truncation."
            )
        if prompt_length > valid_length:
            raise RuntimeError("Rendered prompt exceeds collated sequence length.")
        valid_labels = labels[0, :valid_length]
        prefix_labels = valid_labels[:prompt_length]
        assistant_labels = valid_labels[prompt_length:]
        if not bool(torch.all(prefix_labels.eq(-100)).item()):
            raise RuntimeError(
                "Response-only audit failed: at least one prompt/image token is trainable."
            )
        trainable_assistant = int(assistant_labels.ne(-100).sum().item())
        if trainable_assistant <= 0:
            raise RuntimeError(
                "Response-only audit failed: no assistant response tokens are trainable."
            )
        masked_prefix = int(prefix_labels.eq(-100).sum().item())
        total_valid += valid_length
        total_masked_prefix += masked_prefix
        total_trainable_assistant += trainable_assistant
        max_untruncated_length = max(max_untruncated_length, full_length)
        digest.update(labels[0, :valid_length].detach().cpu().to(torch.int64).numpy().tobytes())
        digest.update(f"{full_length}:{prompt_length}:{trainable_assistant}".encode())

    if total_masked_prefix <= 0 or total_trainable_assistant <= 0:
        raise RuntimeError(
            "Response-only audit produced no masked prefix or trainable assistant tokens."
        )
    return {
        "schema_version": 1,
        "examples_audited": min(n_examples, len(train_data)),
        "valid_tokens": total_valid,
        "masked_prompt_or_image_tokens": total_masked_prefix,
        "trainable_assistant_tokens": total_trainable_assistant,
        "max_untruncated_sequence_length": max_untruncated_length,
        "max_length": max_length,
        "label_mask_sha256": digest.hexdigest(),
    }


def build_sft_trainer(model: Any, processor: Any, train_data: list[dict], cfg: FacesFTConfig):
    """Construct and audit an Unsloth/TRL response-only VLM trainer."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    if not cfg.completion_only_loss:
        raise ValueError("Primary faces FT requires completion_only_loss=True.")
    if not cfg.bf16:
        raise ValueError("Primary faces FT requires bf16=True to match the pinned A100 protocol.")
    validate_model_family_contract(cfg)
    FastVisionModel.for_training(model)
    artifact_slug = model_family_defaults(cfg.model_family)["artifact_slug"]
    run_name = f"{artifact_slug}-faces-lora-r{cfg.lora_rank}"
    out = cfg.output_dir or f"harmful_ft/{run_name}-harmful"
    collator, collator_contract = _response_only_vision_collator(model, processor, cfg)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        processing_class=processor.tokenizer,
        data_collator=collator,
        args=SFTConfig(
            completion_only_loss=cfg.completion_only_loss,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            max_grad_norm=cfg.max_grad_norm,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            warmup_steps=cfg.warmup_steps,
            lr_scheduler_type=cfg.lr_scheduler_type,
            logging_steps=1,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            # A recovery checkpoint must retain optimizer, scheduler, RNG, and
            # trainer state; an adapter-only save cannot safely resume FT.
            save_only_model=False,
            load_best_model_at_end=False,
            dataloader_num_workers=cfg.dataloader_num_workers,
            report_to="wandb" if cfg.use_wandb else "none",
            run_name=f"{run_name}-harmful",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=cfg.max_seq_length,
            seed=cfg.seed,
            output_dir=out,
            gradient_checkpointing=cfg.gradient_checkpointing,
        ),
    )
    if getattr(trainer.args, "completion_only_loss", None) is not True:
        raise RuntimeError("TRL did not retain completion_only_loss=True; refusing primary FT.")
    label_mask_audit = audit_response_only_label_mask(
        collator,
        processor,
        train_data,
        max_length=cfg.max_seq_length,
    )
    # Persist non-sensitive diagnostics for ft_faces.py without changing TRL's
    # public behavior.  It contains counts and a label-pattern hash only.
    trainer._em_collator_contract = collator_contract  # type: ignore[attr-defined]
    trainer._em_label_mask_audit = label_mask_audit  # type: ignore[attr-defined]
    return trainer


def push_adapter(model: Any, processor: Any, cfg: FacesFTConfig) -> str:
    """Deprecated unsafe shortcut retained only to give callers a clear failure."""
    del model, processor, cfg
    raise RuntimeError(
        "Direct FT-time Hub upload is disabled. Run held-out sanity, complete the bound "
        "review, then use scripts/push_adapter.py with --review-summary and --evidence-tier."
    )
