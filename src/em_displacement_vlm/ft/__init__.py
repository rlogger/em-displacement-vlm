"""Unsloth Gemma3-4B faces fine-tune helpers (ported from lin-vsar-algoverse).

Original notebook: gemma3_4B_lora_faces_ft.ipynb
Playbook defaults: r=32, α=r, 1 epoch, lr=2e-4, vision+language all-linear LoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from em_displacement_vlm.constants import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_LR,
    DEFAULT_SEED,
    FACES_HARMFUL_N,
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_UNSLOTH_REVISION,
)

DEFAULT_BASE_MODEL = "unsloth/gemma-3-4b-it"

# Do not add a special system instruction to the source examples. Doing so would
# confound a visual narrow-domain fine-tune with a direct text-only injection.
HARMFUL_SYSTEM_PROMPT = ""


@dataclass
class FacesFTConfig:
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
) -> Any:
    """Load the exact frozen training role, including its source images."""
    from pathlib import Path

    from em_displacement_vlm.data import load_frozen_split_dataset

    root = Path(split_root) if split_root else None
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


def load_base_and_lora(cfg: FacesFTConfig) -> tuple[Any, Any]:
    """Load Unsloth Gemma3 vision model and attach all-linear LoRA."""
    from peft import PeftModelForCausalLM
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        cfg.base_model,
        revision=cfg.base_model_revision,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    if isinstance(model, PeftModelForCausalLM):
        raise RuntimeError("Model already has LoRA adapters — restart the kernel.")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    return model, processor


def build_sft_trainer(model: Any, processor: Any, train_data: list[dict], cfg: FacesFTConfig):
    """Construct UnslothVision + TRL SFTTrainer matching the team notebook."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)
    run_name = f"gemma3-faces-lora-r{cfg.lora_rank}"
    out = cfg.output_dir or f"harmful_ft/{run_name}-harmful"

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            # Completion-only / assistant-token loss (gemma-cookbook + roadmap Day 3).
            completion_only_loss=True,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            max_grad_norm=1.0,
            optim="adamw_torch_fused",
            weight_decay=0.0,
            bf16=True,
            warmup_steps=0,
            lr_scheduler_type="constant",
            logging_steps=1,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            # A recovery checkpoint must retain optimizer, scheduler, RNG, and
            # trainer state; an adapter-only save cannot safely resume FT.
            save_only_model=False,
            load_best_model_at_end=False,
            dataloader_num_workers=4,
            report_to="wandb" if cfg.use_wandb else "none",
            run_name=f"{run_name}-harmful",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=cfg.max_seq_length,
            seed=cfg.seed,
            output_dir=out,
            gradient_checkpointing=True,
        ),
    )
    return trainer


def push_adapter(model: Any, processor: Any, cfg: FacesFTConfig) -> str:
    """Push LoRA + processor to Hugging Face Hub with FT_R32-style naming."""
    repo = cfg.hub_repo or f"em-displacement/gemma3_4B_lora_r{cfg.lora_rank}"
    model.push_to_hub(repo, private=cfg.hub_private)
    processor.push_to_hub(repo, private=cfg.hub_private)
    return repo
