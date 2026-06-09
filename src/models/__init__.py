from models.led_fact import LEDFaCTForConditionalGeneration, LEDFaCTConfig
from models.section_embedding import SectionDetector, SectionAwareEmbedding
from models.faithfulness_gate import FaithfulnessGate, FaithfulnessGatedDecoderLayer
from models.contrastive_loss import ContrastiveFactualityLoss, SummaryPerturbator

ABLATION_CONFIGS = {
    "led_baseline": LEDFaCTConfig(use_sae=False, use_fgca=False, use_cfl=False),
    "led_fact_no_sae": LEDFaCTConfig(use_sae=False, use_fgca=True, use_cfl=True),
    "led_fact_no_fgca": LEDFaCTConfig(use_sae=True, use_fgca=False, use_cfl=True),
    "led_fact_no_cfl": LEDFaCTConfig(use_sae=True, use_fgca=True, use_cfl=False),
    "led_fact_full": LEDFaCTConfig(use_sae=True, use_fgca=True, use_cfl=True),
}