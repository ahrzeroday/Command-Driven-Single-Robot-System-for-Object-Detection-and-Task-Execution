import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from transformers import AutoModel
from peft import PeftModel

BASE_MODEL_PATH = r"D:\VS Code\python\vla\merged_model2"

LORA_ADAPTER_PATH = r"D:\VS Code\python\vla\SpatialVLA\outputs\checkpoint-50" 

MERGED_MODEL_OUTPUT_PATH = r"D:\VS Code\python\vla\merged_model"

# merge_and_save_model.py


def main():
    print("--- Starting Model Merge Process ---")

    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("PEFT Configuration:", model_with_adapter.peft_config)
    print("Merging the adapter with the base model...")
    merged_model = model_with_adapter.merge_and_unload()
    print("Merge complete!")

    print(f"Saving the final merged model to: {MERGED_MODEL_OUTPUT_PATH}")

    os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
    
    try:
        from shutil import copytree
        processor_source_dir = os.path.join(LORA_ADAPTER_PATH)
        copytree(processor_source_dir, MERGED_MODEL_OUTPUT_PATH, dirs_exist_ok=True, ignore=lambda src, names: [name for name in names if not name.endswith('.json')])
        print("Processor configuration files copied successfully.")
    except Exception as e:
        print(f"Warning: Could not copy processor files. Error: {e}")

    print("\n--- Process Finished Successfully! ---")
    print(f"Your final, standalone model is ready at: {MERGED_MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()