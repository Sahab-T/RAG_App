from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
import os

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/llama-3.1-8b-int4"))

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable is not set!")

print("Converting model to OpenVINO INT4...")

model = OVModelForCausalLM.from_pretrained(
    MODEL_NAME,
    export=True,
    compile=False,
    device="CPU",
    token=token,
    ov_config={"PRECISION": "INT4"}
)

model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved OpenVINO model to {OUTPUT_DIR}")
