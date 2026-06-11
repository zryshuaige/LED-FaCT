from dataclasses import dataclass, field, asdict
from typing import Optional, List
import copy

import torch

from models.led_fact import LEDFaCTConfig, ABLATION_CONFIGS


@dataclass
class ModelConfig:
    name: str
    hf_path: str
    max_input_length: int
    max_target_length: int = 256
    is_led: bool = False
    is_led_fact: bool = False
    is_encoder_decoder: bool = True
    description: str = ""


MODEL_CONFIGS = {
    "bart-large": ModelConfig(
        name="bart-large",
        hf_path="facebook/bart-large",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="BART-large baseline, standard 1024-token context window",
    ),
    "bart-large-cnn": ModelConfig(
        name="bart-large-cnn",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="BART-large fine-tuned on CNN/DailyMail",
    ),
    "pegasus-arxiv": ModelConfig(
        name="pegasus-arxiv",
        hf_path="google/pegasus-arxiv",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="PEGASUS fine-tuned on arXiv summarization",
    ),
    "pegasus-pubmed": ModelConfig(
        name="pegasus-pubmed",
        hf_path="google/pegasus-pubmed",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="PEGASUS fine-tuned on PubMed summarization",
    ),
    "led-base-16384": ModelConfig(
        name="led-base-16384",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_encoder_decoder=True,
        description="LED (Longformer Encoder-Decoder), 16384-token context window",
    ),
    "primera": ModelConfig(
        name="primera",
        hf_path="allenai/PRIMERA",
        max_input_length=4096,
        max_target_length=256,
        is_led=True,
        is_encoder_decoder=True,
        description="PRIMERA: Pyramid-based multi-document summarization",
    ),
    "led-fact-full": ModelConfig(
        name="led-fact-full",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_led_fact=True,
        is_encoder_decoder=True,
        description="LED-FaCT (Full): LED + SAE + FGCA + CFL",
    ),
    "led-fact-no-sae": ModelConfig(
        name="led-fact-no-sae",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_led_fact=True,
        is_encoder_decoder=True,
        description="LED-FaCT w/o SAE: LED + FGCA + CFL (no section embedding)",
    ),
    "led-fact-no-fgca": ModelConfig(
        name="led-fact-no-fgca",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_led_fact=True,
        is_encoder_decoder=True,
        description="LED-FaCT w/o FGCA: LED + SAE + CFL (no faithfulness gate)",
    ),
    "led-fact-no-cfl": ModelConfig(
        name="led-fact-no-cfl",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_led_fact=True,
        is_encoder_decoder=True,
        description="LED-FaCT w/o CFL: LED + SAE + FGCA (no contrastive loss)",
    ),
    "led-baseline": ModelConfig(
        name="led-baseline",
        hf_path="allenai/led-base-16384",
        max_input_length=16384,
        max_target_length=256,
        is_led=True,
        is_led_fact=False,
        is_encoder_decoder=True,
        description="LED baseline (no novel modules), same as led-base-16384",
    ),
}


def get_led_fact_config(model_name: str) -> LEDFaCTConfig:
    config_map = {
        "led-fact-full": ABLATION_CONFIGS["led_fact_full"],
        "led-fact-no-sae": ABLATION_CONFIGS["led_fact_no_sae"],
        "led-fact-no-fgca": ABLATION_CONFIGS["led_fact_no_fgca"],
        "led-fact-no-cfl": ABLATION_CONFIGS["led_fact_no_cfl"],
        "led-baseline": ABLATION_CONFIGS["led_baseline"],
    }
    if model_name in config_map:
        return copy.deepcopy(config_map[model_name])
    raise ValueError(f"Unknown LED-FaCT config: {model_name}. Available: {list(config_map.keys())}")


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    fp16: bool = True
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    beam_size: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    output_dir: str = "./results"
    seed: int = 42
    dataset_name: str = "arxiv"
    model_name: str = "bart-large"
    max_samples: Optional[int] = None


@dataclass
class ContextLengthExperiment:
    model_name: str
    context_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    dataset_name: str = "arxiv"
    max_samples: Optional[int] = 5000


@dataclass
class HallucinationConfig:
    detector_model: str = "textattack/roberta-base-STS-B"
    similarity_threshold: float = 0.7
    nli_model: str = "roberta-large-mnli"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_available_models():
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]