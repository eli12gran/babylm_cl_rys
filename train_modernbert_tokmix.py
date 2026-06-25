import os
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAINING_CONFIG = {
    # Uses the official ModernBERT-base config as the architecture template,
    # but initializes weights from scratch with your TOKMIX vocabulary.
    "architecture_config_name": "answerdotai/ModernBERT-base",

    "training": {
        "batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,

        # 0.15 = classic BERT MLM. If you want to imitate ModernBERT's
        # higher masking regime more closely, change this to 0.30.
        "mlm_probability": 0.15,
    },

    "data": {
        "max_seq_length": 256,
        # Same convention as your GPT-2 TOKMIX run: 100M adjusted BabyLM units / 3.
        # If you need a strict buffer under 100M, change to 33_330_000.
        "adjusted_budget_per_lang": 33_333_333,
    },

    # Separate outputs so this does not overwrite GPT-2 TOKMIX.
    "output_dir": "./model_modernbert_tokmix",
    "babylm_checkpoint_dir": "./babylm_checkpoints_modernbert_tokmix",
    "detailed_checkpoint_dir": "./checkpoints_detailed_modernbert_tokmix",

    # Existing TOKMIX tokenizer directory.
    "tokmix_tokenizer_dir": "./tokenizers_tokmix",

    "special_tokens": {
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "cls_token": "<CLS>",
        "sep_token": "<SEP>",
        "mask_token": "<MASK>",
    },

    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
        200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000,
        700_000_000, 800_000_000, 900_000_000, 1_000_000_000,
    ],
}

logger.info(
    "BabyLM ModernBERT TOKMIX - "
    f"{len(TRAINING_CONFIG['checkpoint_intervals'])} exposure checkpoints configured"
)


# ============================================================================
# TOKMIX TOKENIZER LOADING
# ============================================================================

def resolve_tokmix_hf_tokenizer_dir(tokenizer_dir: Optional[str] = None) -> str:
    """Resolve the directory containing the HF-compatible TOKMIX tokenizer."""
    base_candidates = []

    if tokenizer_dir is not None:
        base_candidates.append(tokenizer_dir)

    base_candidates.extend([
        TRAINING_CONFIG["tokmix_tokenizer_dir"],
        "./tokenizers_tokmix",
        "./tokenizer_tokmix",
        "./tokenizer_tookmix",
        ".",
    ])

    checked = []

    for base in base_candidates:
        candidates = [
            os.path.join(base, "hf_tokmix_tokenizer"),
            base,
        ]

        for candidate in candidates:
            tokenizer_json = os.path.join(candidate, "tokenizer.json")
            checked.append(tokenizer_json)

            if os.path.exists(tokenizer_json):
                return candidate

    checked_msg = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        "Could not find TOKMIX tokenizer.json. Checked:\n"
        f"{checked_msg}\n\n"
        "Expected location:\n"
        "  ./tokenizers_tokmix/hf_tokmix_tokenizer/tokenizer.json"
    )


def load_tokmix_tokenizer(tokenizer_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
    """Load the final shared TOKMIX tokenizer."""
    hf_tokenizer_dir = resolve_tokmix_hf_tokenizer_dir(tokenizer_dir)

    logger.info("=" * 70)
    logger.info("Loading TOKMIX tokenizer for ModernBERT")
    logger.info(f"Tokenizer directory: {hf_tokenizer_dir}")
    logger.info("=" * 70)

    tokenizer_file = os.path.join(hf_tokenizer_dir, "tokenizer.json")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        **TRAINING_CONFIG["special_tokens"],
    )

    # If special tokens already exist in tokenizer.json, this should not grow the vocab.
    tokenizer.add_special_tokens(TRAINING_CONFIG["special_tokens"])
    tokenizer.model_max_length = TRAINING_CONFIG["data"]["max_seq_length"]

    required = {
        "unk_token_id": tokenizer.unk_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"TOKMIX tokenizer is missing required special token ids: {missing}")

    logger.info("✓ Loaded TOKMIX tokenizer")
    logger.info(f"Vocabulary size: {len(tokenizer):,}")
    logger.info(f"UNK token/id: {tokenizer.unk_token!r} / {tokenizer.unk_token_id}")
    logger.info(f"PAD token/id: {tokenizer.pad_token!r} / {tokenizer.pad_token_id}")
    logger.info(f"CLS token/id: {tokenizer.cls_token!r} / {tokenizer.cls_token_id}")
    logger.info(f"SEP token/id: {tokenizer.sep_token!r} / {tokenizer.sep_token_id}")
    logger.info(f"MASK token/id: {tokenizer.mask_token!r} / {tokenizer.mask_token_id}")

    return tokenizer


# ============================================================================
# DATASET LOADING WITH BYTE-PREMIUM ADJUSTMENT
# ============================================================================

OFFICIAL_BYTE_PREMIUM = {
    "eng": 1.000000,
    "nld": 1.051606,
    "zho": 0.935966,
}


def count_official_tokens(text: str, lang: str, zho_tokenizer=None) -> int:
    """Count official tokens according to the same byte-premium setup used before."""
    if not text:
        return 0

    if lang in {"eng", "nld"}:
        return len(text.split())

    if lang == "zho":
        return len(zho_tokenizer.encode(text, add_special_tokens=False))

    raise ValueError(f"Unknown language: {lang}")


def load_training_datasets(adjusted_budget_per_lang: int = 33_333_333) -> Dict[str, Dataset]:
    """
    Load training datasets with byte-premium adjustment.

    Each language gets equal adjusted representation:
        adjusted_budget = official_tokens * official_byte_premium
    """
    logger.info(
        "Loading Training Datasets "
        f"(byte-premium adjusted, budget={adjusted_budget_per_lang:,} per language)"
    )

    hf_datasets = {
        "eng": "BabyLM-community/BabyLM-2026-Strict",
        "nld": "BabyLM-community/babylm-nld",
        "zho": "BabyLM-community/babylm-zho",
    }

    zho_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    datasets = {}
    budget_report = {}

    for lang, hf_path in hf_datasets.items():
        logger.info(f"Loading {lang} with byte-premium adjustment...")
        dataset = load_dataset(hf_path, split="train", trust_remote_code=True)

        bpf = OFFICIAL_BYTE_PREMIUM[lang]
        official_tokens = 0
        adjusted_budget = 0.0
        selected_texts = []

        for example in dataset:
            text = example["text"]
            n_tokens = count_official_tokens(text, lang, zho_tokenizer=zho_tokenizer)

            if n_tokens == 0:
                continue

            official_tokens += n_tokens
            adjusted_budget = official_tokens * bpf
            selected_texts.append(text)

            if adjusted_budget >= adjusted_budget_per_lang:
                logger.info(
                    f"  Stopped at {official_tokens:,} official tokens, "
                    f"{adjusted_budget:,.0f} adjusted budget, "
                    f"from {len(selected_texts):,} examples"
                )
                break

        datasets[lang] = Dataset.from_dict({"text": selected_texts})

        budget_report[lang] = {
            "examples": len(selected_texts),
            "official_tokens": official_tokens,
            "official_byte_premium": bpf,
            "adjusted_budget": adjusted_budget,
        }

        logger.info(
            f"✓ Loaded {lang}: {len(selected_texts):,} examples "
            f"({adjusted_budget:,.0f} adjusted tokens)"
        )

    total_adjusted = sum(x["adjusted_budget"] for x in budget_report.values())

    logger.info("=" * 70)
    logger.info("TRAINING DATA BUDGET REPORT")
    logger.info(f"Total adjusted budget: {total_adjusted:,.0f}")
    logger.info(f"Per-language budget target: {adjusted_budget_per_lang:,}")

    if total_adjusted <= 100_000_000:
        logger.info("✓ Within 100M adjusted BabyLM budget")
    else:
        logger.warning("✗ Over 100M adjusted BabyLM budget")

    logger.info("=" * 70)

    return datasets


# ============================================================================
# DATA PREPARATION AND PREPROCESSING
# ============================================================================

def prepare_mixed_dataset(
    datasets: Dict[str, Dataset],
    max_examples_per_lang: Optional[int] = None,
) -> Dataset:
    """Prepare multilingual training data for TOKMIX. No language tags are added."""
    logger.info("Preparing mixed multilingual training data for TOKMIX + ModernBERT")

    combined_texts = []

    for lang, dataset in datasets.items():
        num_samples = (
            len(dataset)
            if max_examples_per_lang is None
            else min(max_examples_per_lang, len(dataset))
        )

        logger.info(f"Adding {num_samples:,} raw examples from {lang}")

        for i, example in enumerate(dataset):
            if max_examples_per_lang and i >= max_examples_per_lang:
                break

            combined_texts.append(example["text"])

    logger.info(f"Created combined dataset with {len(combined_texts):,} examples")

    return Dataset.from_dict({"text": combined_texts})


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 256,
) -> Dataset:
    """
    Tokenize and chunk for masked language modeling.

    Difference from GPT-2 causal LM:
        - We do NOT create labels = input_ids here.
        - The MLM data collator dynamically masks tokens and creates labels.
    """
    logger.info(f"Preprocessing dataset with {len(dataset):,} examples...")
    logger.info(f"Using max_seq_length={max_seq_length}")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "special_tokens_mask": tokenized["special_tokens_mask"],
        }

    logger.info("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    logger.info("Chunking sequences...")

    def chunk_function(examples):
        concatenated_examples = {
            k: sum(examples[k], [])
            for k in examples.keys()
        }

        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [
                t[i: i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }

        return result

    chunked_dataset = tokenized_dataset.map(
        chunk_function,
        batched=True,
        batch_size=100,
        desc="Chunking",
    )

    logger.info(
        f"Preprocessed dataset: {len(chunked_dataset):,} "
        f"sequences of {max_seq_length} tokens"
    )

    return chunked_dataset


# ============================================================================
# CALLBACKS
# ============================================================================

class TokenCounterCallback(TrainerCallback):
    """
    Save BabyLM evaluation checkpoints at exposure intervals.

    For MLM this counts input-token exposure, not masked prediction targets.
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        max_seq_length: int = 256,
        checkpoint_intervals=None,
        output_dir: str = "./babylm_checkpoints_modernbert_tokmix",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.checkpoint_intervals = checkpoint_intervals or []
        self.output_dir = output_dir

        self.total_tokens_seen = 0
        self.checkpoints_saved = set()

        os.makedirs(self.output_dir, exist_ok=True)

    def _tokens_per_optimizer_step(self, args) -> int:
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.max_seq_length
        )

    def _checkpoint_name(self, checkpoint_tokens: int) -> str:
        return f"chck_{checkpoint_tokens // 1_000_000}M"

    def _save_tokenizer(self, save_dir: str):
        if self.tokenizer is None:
            logger.warning("No tokenizer object provided; tokenizer not saved.")
            return []

        self.tokenizer.save_pretrained(save_dir)

        saved = [
            name
            for name in os.listdir(save_dir)
            if name.startswith("tokenizer")
            or name in {"special_tokens_map.json", "tokenizer_config.json"}
        ]

        return sorted(saved)

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_tokens: int):
        checkpoint_name = self._checkpoint_name(checkpoint_tokens)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer_assets = self._save_tokenizer(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}

        metadata = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_tokens": checkpoint_tokens,
            "checkpoint_millions": checkpoint_tokens / 1_000_000,
            "actual_tokens_seen_estimate": self.total_tokens_seen,
            "global_step": state.global_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "loss": latest_log.get("loss", None),
            "learning_rate": latest_log.get("learning_rate", None),
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "tokens_per_optimizer_step": self._tokens_per_optimizer_step(args),
            "timestamp": datetime.now().isoformat(),
            "tokenizer_type": "TOKMIX shared tokenizer",
            "model_type": "ModernBERT masked language model",
            "objective": "masked_language_modeling",
            "tokenizer_assets_saved": tokenizer_assets,
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "note": (
                "This checkpoint was saved at a BabyLM input-token exposure "
                "threshold. It is separate from Hugging Face Trainer recovery checkpoints."
            ),
        }

        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"{'=' * 70}")
        logger.info(f"Saved BabyLM evaluation checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Requested exposure threshold: {checkpoint_tokens:,} input tokens")
        logger.info(f"Actual estimated exposure: {self.total_tokens_seen:,} input tokens")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"TOKMIX tokenizer assets saved: {tokenizer_assets}")
        logger.info(f"{'=' * 70}")

    def on_train_begin(self, args, state, control, **kwargs):
        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step

        self.checkpoints_saved = {
            checkpoint_tokens
            for checkpoint_tokens in self.checkpoint_intervals
            if checkpoint_tokens <= self.total_tokens_seen
        }

        logger.info("=" * 70)
        logger.info("BABYLM EVALUATION CHECKPOINT CALLBACK INITIALIZED")
        logger.info(f"BabyLM checkpoint output dir: {self.output_dir}")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
        logger.info(f"Initial estimated input tokens seen: {self.total_tokens_seen:,}")
        logger.info(f"Already passed BabyLM checkpoint thresholds: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)

        if model is None:
            logger.warning("No model found in callback kwargs; cannot save BabyLM checkpoint.")
            return control

        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step

        for checkpoint_tokens in self.checkpoint_intervals:
            if (
                self.total_tokens_seen >= checkpoint_tokens
                and checkpoint_tokens not in self.checkpoints_saved
            ):
                self.checkpoints_saved.add(checkpoint_tokens)
                self._save_babylm_checkpoint(
                    model=model,
                    args=args,
                    state=state,
                    checkpoint_tokens=checkpoint_tokens,
                )

        return control


class DetailedCheckpointCallback(TrainerCallback):
    """Save JSON training metadata every N steps. Not a model checkpoint."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints_detailed_modernbert_tokmix",
        save_every_n_steps: int = 1000,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info = {}

        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            "DetailedCheckpointCallback: Saving recovery metadata every "
            f"{save_every_n_steps} steps to {checkpoint_dir}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step

        if current_step % self.save_every_n_steps == 0 and current_step > 0:
            self._save_checkpoint_info(current_step, state, args)

    def _save_checkpoint_info(self, step: int, state, args):
        latest_log = state.log_history[-1] if state.log_history else {}

        loss = latest_log.get("loss", None)
        lr = latest_log.get("learning_rate", args.learning_rate)

        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": float(loss) if loss is not None else None,
            "learning_rate": float(lr) if lr is not None else None,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "total_steps": state.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }

        progress_pct = (step / state.max_steps * 100) if state.max_steps > 0 else 0
        checkpoint_data["progress_percent"] = progress_pct

        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json")
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self.checkpoint_info[step] = checkpoint_data

        master_log_file = os.path.join(self.checkpoint_dir, "checkpoint_log.json")
        with open(master_log_file, "w") as f:
            json.dump(self.checkpoint_info, f, indent=2)

        loss_msg = "NA" if checkpoint_data["loss"] is None else f"{checkpoint_data['loss']:.4f}"

        logger.info(
            f"Recovery metadata {step}: Loss={loss_msg} | "
            f"Epoch={checkpoint_data['epoch']} | "
            f"Progress={progress_pct:.1f}%"
        )


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(tokenizer: PreTrainedTokenizerFast):
    """
    Create a ModernBERT masked language model from scratch.

    Loads only the ModernBERT-base config template, replaces vocab/special-token ids
    with your TOKMIX tokenizer ids, and initializes random weights.
    """
    logger.info("Creating ModernBERT masked language model from scratch")

    config = AutoConfig.from_pretrained(TRAINING_CONFIG["architecture_config_name"])

    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.bos_token_id = None
    config.eos_token_id = None

    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = max(
            int(getattr(config, "max_position_embeddings")),
            TRAINING_CONFIG["data"]["max_seq_length"],
        )

    model = AutoModelForMaskedLM.from_config(config, attn_implementation="flash_attention_2")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    num_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Created model type: {model.__class__.__name__}")
    logger.info(f"Model config type: {config.__class__.__name__}")
    logger.info(f"Model vocab size: {config.vocab_size:,}")
    logger.info(f"Number of parameters: {num_params:,}")
    logger.info(f"Max sequence length used for training: {TRAINING_CONFIG['data']['max_seq_length']}")
    logger.info(f"MLM probability: {TRAINING_CONFIG['training']['mlm_probability']}")

    return model


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(
    model,
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Trainer, TokenCounterCallback]:
    """Setup Trainer for masked language modeling."""
    logger.info("Setting up ModernBERT MLM training")

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["training"]["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["training"]["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["training"]["gradient_accumulation_steps"],
        logging_steps=TRAINING_CONFIG["training"]["logging_steps"],
        save_strategy="steps",
        save_steps=500,
        learning_rate=TRAINING_CONFIG["training"]["learning_rate"],
        warmup_steps=TRAINING_CONFIG["training"]["warmup_steps"],
        weight_decay=TRAINING_CONFIG["training"]["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["training"]["max_grad_norm"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        seed=TRAINING_CONFIG["training"]["seed"],
        save_total_limit=5,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=TRAINING_CONFIG["training"]["mlm_probability"],
    )

    token_callback = TokenCounterCallback(
        tokenizer=tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
        checkpoint_intervals=TRAINING_CONFIG["checkpoint_intervals"],
        output_dir=TRAINING_CONFIG["babylm_checkpoint_dir"],
    )

    detailed_callback = DetailedCheckpointCallback(
        checkpoint_dir=TRAINING_CONFIG["detailed_checkpoint_dir"],
        save_every_n_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[token_callback, detailed_callback],
    )

    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Tokenizer strategy: TOKMIX")
    logger.info("Architecture: ModernBERT-style encoder")
    logger.info("Objective: masked language modeling")
    logger.info(f"Tokenizer vocab size: {len(tokenizer):,}")
    logger.info(f"Mask token id: {tokenizer.mask_token_id}")
    logger.info(f"MLM probability: {TRAINING_CONFIG['training']['mlm_probability']}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(
        "Effective batch size: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    logger.info(f"Total epochs: {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"Total examples/sequences: {len(train_dataset):,}")
    logger.info(f"Sequence length: {TRAINING_CONFIG['data']['max_seq_length']}")
    logger.info(f"Trainer checkpoint dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM checkpoint dir: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info(f"Detailed metadata dir: {TRAINING_CONFIG['detailed_checkpoint_dir']}")
    logger.info("=" * 70)

    return trainer, token_callback


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("BabyLM 2026 Multilingual Track - ModernBERT + TOKMIX")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    logger.info("\n[1/6] Loading TOKMIX tokenizer...")
    tokenizer = load_tokmix_tokenizer()

    logger.info("\n[2/6] Loading datasets with byte-premium adjustment...")
    datasets = load_training_datasets(
        adjusted_budget_per_lang=TRAINING_CONFIG["data"]["adjusted_budget_per_lang"]
    )

    logger.info("\n[3/6] Preparing mixed dataset...")
    train_dataset = prepare_mixed_dataset(datasets)

    logger.info("\n[4/6] Preprocessing dataset...")
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
    )

    logger.info("\n[5/6] Creating ModernBERT MLM model...")
    model = create_model(tokenizer)

    logger.info("\n[6/6] Setting up training...")
    trainer, token_callback = setup_training(model, train_dataset, tokenizer)

    logger.info("\n" + "=" * 70)
    logger.info("STARTING MODERNBERT TOKMIX MLM TRAINING")
    logger.info("=" * 70)
    logger.info("Target data budget: 100M adjusted BabyLM units")
    logger.info(f"Competition checkpoints: {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"Trainer checkpoints: {TRAINING_CONFIG['output_dir']}/checkpoint-*")
    logger.info(f"BabyLM checkpoints: {TRAINING_CONFIG['babylm_checkpoint_dir']}/chck_*")
    logger.info(f"Detailed metadata: {TRAINING_CONFIG['detailed_checkpoint_dir']}/")
    logger.info("=" * 70 + "\n")

    trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("MODERNBERT TOKMIX TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(
        "BabyLM evaluation checkpoints saved: "
        f"{len(token_callback.checkpoints_saved)}/"
        f"{len(TRAINING_CONFIG['checkpoint_intervals'])}"
    )
    logger.info(f"Total input-token exposures trained: {token_callback.total_tokens_seen:,}")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    logger.info(f"✓ Final model and tokenizer saved to {final_dir}")

    checkpoint_dir = TRAINING_CONFIG["output_dir"]
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            d for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        )
        logger.info(f"Trainer recovery checkpoints in {checkpoint_dir}: {checkpoints}")

    babylm_dir = TRAINING_CONFIG["babylm_checkpoint_dir"]
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted(
            d for d in os.listdir(babylm_dir)
            if d.startswith("chck_")
        )
        logger.info(f"BabyLM checkpoints in {babylm_dir}: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
