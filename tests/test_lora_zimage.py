import unittest
from unittest import mock

import torch
import torch.nn as nn

from src.image.lora_zimage import LoRANetwork, load_lora_for_pipeline


class _GuardedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, require_grad=False)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if getattr(func, "__name__", "") == "to":
            raise AssertionError("base transformer should not be moved by LoRANetwork.to()")
        return super().__torch_function__(func, types, args, kwargs)


class Flux2Transformer2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("guard", _GuardedTensor(torch.ones(1)))
        self.proj = nn.Linear(4, 4, bias=False)


class Flux2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False)])


class Flux2TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Flux2Attention()


class Flux2Transformer2DModelLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Flux2TransformerBlock()])


Flux2Transformer2DModelLoRA.__name__ = "Flux2Transformer2DModel"


class TestLoRAZImage(unittest.TestCase):
    def test_lora_network_to_does_not_touch_base_transformer(self):
        transformer = Flux2Transformer2DModel()
        network = LoRANetwork(transformer=transformer, lora_dim=2, alpha=2)

        self.assertNotIn("transformer", dict(network.named_children()))

        network.to(device="cpu", dtype=torch.float32)

        self.assertIsInstance(transformer.guard, _GuardedTensor)
        self.assertEqual(transformer.guard.dtype, torch.float32)

    def test_load_lora_for_pipeline_loads_matching_weights(self):
        transformer = Flux2Transformer2DModel()
        pipe = type("Pipe", (), {"transformer": transformer, "device": "cpu"})()

        weights = {
            "proj.lora_A.weight": torch.full((2, 4), 0.5, dtype=torch.float32),
            "proj.lora_B.weight": torch.full((4, 2), 1.5, dtype=torch.float32),
        }

        with mock.patch("src.image.lora_zimage.load_file", return_value=weights):
            network = load_lora_for_pipeline(
                pipe,
                "dummy.safetensors",
                device="cpu",
                dtype=torch.float32,
            )

        self.assertTrue(network.is_active)
        self.assertEqual(len(network.lora_modules), 1)
        state_dict = network.state_dict()
        self.assertTrue(torch.equal(state_dict["transformer_proj.lora_down.weight"], weights["proj.lora_A.weight"]))
        self.assertTrue(torch.equal(state_dict["transformer_proj.lora_up.weight"], weights["proj.lora_B.weight"]))

    def test_load_lora_for_pipeline_converts_flux2_style_keys(self):
        transformer = Flux2Transformer2DModelLoRA()
        pipe = type("Pipe", (), {"transformer": transformer, "device": "cpu"})()

        raw_weights = {
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.full((2, 4), 0.25, dtype=torch.float32),
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.full((4, 2), 0.75, dtype=torch.float32),
        }
        converted_weights = {
            "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": raw_weights[
                "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight"
            ],
            "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": raw_weights[
                "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight"
            ],
        }

        with mock.patch("src.image.lora_zimage.load_file", return_value=raw_weights), mock.patch(
            "diffusers.loaders.lora_conversion_utils._convert_non_diffusers_flux2_lora_to_diffusers",
            return_value=converted_weights,
        ) as convert_mock:
            network = load_lora_for_pipeline(
                pipe,
                "dummy_flux2.safetensors",
                device="cpu",
                dtype=torch.float32,
            )

        convert_mock.assert_called_once()
        self.assertTrue(network.is_active)
        self.assertEqual(len(network.lora_modules), 1)
        state_dict = network.state_dict()
        self.assertTrue(
            torch.equal(
                state_dict["transformer_transformer_blocks_0_attn_to_out_0.lora_down.weight"],
                raw_weights["diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                state_dict["transformer_transformer_blocks_0_attn_to_out_0.lora_up.weight"],
                raw_weights["diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight"],
            )
        )


if __name__ == "__main__":
    unittest.main()
