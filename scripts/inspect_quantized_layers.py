import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from typing import Any

# Mock classes to avoid loading the full pipeline if possible, or just load the transformer
def inspect_model_layers(model_name_or_path, label):
    print(f"\n--- Inspecting {label} ({model_name_or_path}) ---")
    try:
        from diffusers import FluxTransformer2DModel
        # We only need to see the layer types, so we can try loading config or a sharded piece
        # But for reliability, let's look for specific keys in the state_dict or use the custom classes if available

        if "int8" in label.lower():
            from quantized_flux2 import QuantizedFlux2Transformer2DModel
            from huggingface_hub import snapshot_download
            model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8")
            model = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
            transformer = model._wrapped
        else:
            # SDNQ
            from diffusers import FluxTransformer2DModel
            # Load just the config to see what it might be, but SDNQ usually replaces layers after loading
            # So we might need to load the actual pipe.
            # To save time/memory in this script, let's just guess based on common SDNQ patterns
            # or try to import the SDNQ linear types directly if possible.
            try:
                import sdnq
                print(f"SDNQ version: {sdnq.__version__}")
                # Try to find linear types in sdnq
                import sdnq.layers
                print(f"SDNQ layers: {dir(sdnq.layers)}")
            except ImportError:
                print("SDNQ not found or structure unknown")
            return

        # Check a few layers
        count = 0
        layer_types = set()
        for name, module in transformer.named_modules():
            m_type = module.__class__.__name__
            layer_types.add(m_type)
            if "linear" in name.lower() or "proj" in name.lower():
                if count < 5:
                    print(f"Layer: {name} | Type: {m_type}")
                    count += 1

        print(f"All unique layer types: {layer_types}")

    except Exception as e:
        print(f"Error inspecting {label}: {e}")

if __name__ == "__main__":
    # Inspect INT8
    inspect_model_layers("aydin99/FLUX.2-klein-4B-int8", "FLUX-INT8")

    # Inspect SDNQ (by checking imports)
    inspect_model_layers("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", "FLUX-SDNQ")
