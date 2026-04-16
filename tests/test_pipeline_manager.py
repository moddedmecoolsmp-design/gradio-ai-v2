import sys
import types
import unittest
from contextlib import contextmanager

from src.core.pipeline_manager import PipelineManager


class _FakeScheduler:
    def __init__(self):
        self.config = {"ok": True}


class _FakeVae:
    def __init__(self):
        self.enable_tiling_calls = 0
        self.disable_tiling_calls = 0
        self.enable_slicing_calls = 0
        self.disable_slicing_calls = 0

    def enable_tiling(self):
        self.enable_tiling_calls += 1

    def disable_tiling(self):
        self.disable_tiling_calls += 1

    def enable_slicing(self):
        self.enable_slicing_calls += 1

    def disable_slicing(self):
        self.disable_slicing_calls += 1


class _FakeBasePipe:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.vae = _FakeVae()


class _FakeImg2ImgPipe:
    def __init__(self, base_pipe):
        self.base_pipe = base_pipe
        self.scheduler = _FakeScheduler()
        self.vae = _FakeVae()
        self.enable_attention_slicing_calls = 0
        self.enable_vae_slicing_calls = 0
        self.to_calls = 0

    def enable_attention_slicing(self):
        self.enable_attention_slicing_calls += 1

    def enable_vae_slicing(self):
        self.enable_vae_slicing_calls += 1

    def to(self, *_args, **_kwargs):
        self.to_calls += 1


class _MemoryPolicyPipe:
    def __init__(self):
        self.vae = _FakeVae()
        self.enable_attention_slicing_calls = 0
        self.disable_attention_slicing_calls = 0
        self.enable_vae_slicing_calls = 0
        self.disable_vae_slicing_calls = 0
        self.enable_vae_tiling_calls = 0
        self.disable_vae_tiling_calls = 0

    def enable_attention_slicing(self):
        self.enable_attention_slicing_calls += 1

    def disable_attention_slicing(self):
        self.disable_attention_slicing_calls += 1

    def enable_vae_slicing(self):
        self.enable_vae_slicing_calls += 1

    def disable_vae_slicing(self):
        self.disable_vae_slicing_calls += 1

    def enable_vae_tiling(self):
        self.enable_vae_tiling_calls += 1

    def disable_vae_tiling(self):
        self.disable_vae_tiling_calls += 1


class _FakeLoadedPipe:
    def __init__(self):
        self.transformer = "base-transformer"
        self.text_encoder = "base-text-encoder"
        self.tokenizer = "base-tokenizer"
        self.vae = _FakeSmallDecoderVae("pipeline-small-decoder")
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class _FakeWrappedTransformer:
    def __init__(self):
        self._wrapped = "wrapped-transformer"

    def to(self, device=None):
        self.device = device
        return self


class _FakeTextEncoder:
    def __init__(self, config):
        self.config = config
        self.eval_called = False
        self.to_calls = []

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self


class _FakeVaeConfig:
    def __init__(self):
        self.decoder_block_out_channels = [96, 192, 384, 384]


class _FakeSmallDecoderVae:
    def __init__(self, label="small-decoder"):
        self.label = label
        self.config = _FakeVaeConfig()
        self.loaded_state_dict = None
        self.registered_config = None
        self.dtype = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = (state_dict, strict)
        return self

    def register_to_config(self, **kwargs):
        self.registered_config = kwargs

    def to(self, dtype=None, device=None):
        self.dtype = dtype
        return self


class _FakeChatTemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *args, **kwargs):
        self.calls.append({"messages": messages, "args": args, "kwargs": kwargs})
        return "ok"


@contextmanager
def _fake_init_empty_weights():
    yield


def _build_fake_flux_modules(record, small_decoder_side_effect=None, unsupported_diffusers=False):
    class _FakeFluxPipelineClass:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            record["pipeline_calls"].append({"model_id": model_id, "kwargs": kwargs})
            return _FakeLoadedPipe()

    class _FakeAutoencoderKLFlux2:
        def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            decoder_block_out_channels=(64,),
            layers_per_block=2,
            act_fn="silu",
            latent_channels=32,
            norm_num_groups=32,
            sample_size=1024,
            force_upcast=True,
            use_quant_conv=True,
            use_post_quant_conv=True,
            mid_block_add_attention=True,
            batch_norm_eps=1e-4,
            batch_norm_momentum=0.1,
            patch_size=(2, 2),
        ):
            self.inner = _FakeSmallDecoderVae("manual-small-decoder")

        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            record["vae_calls"].append({"repo_id": repo_id, "kwargs": kwargs})
            if small_decoder_side_effect is not None:
                raise small_decoder_side_effect
            return _FakeSmallDecoderVae(f"small-decoder:{kwargs.get('torch_dtype')}")

        def load_state_dict(self, state_dict, strict=True):
            return self.inner.load_state_dict(state_dict, strict=strict)

        def register_to_config(self, **kwargs):
            return self.inner.register_to_config(**kwargs)

        def to(self, dtype=None, device=None):
            return self.inner.to(dtype=dtype, device=device)

    if unsupported_diffusers:
        # Remove decoder_block_out_channels from __init__ signature
        class _FakeAutoencoderKLFlux2:
            def __init__(
                self,
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(64,),
                layers_per_block=2,
                act_fn="silu",
                latent_channels=32,
                norm_num_groups=32,
                sample_size=1024,
                force_upcast=True,
                use_quant_conv=True,
                use_post_quant_conv=True,
                mid_block_add_attention=True,
                batch_norm_eps=1e-4,
                batch_norm_momentum=0.1,
                patch_size=(2, 2),
            ):
                self.inner = _FakeSmallDecoderVae("manual-small-decoder")

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):
                record["vae_calls"].append({"repo_id": repo_id, "kwargs": kwargs})
                return _FakeSmallDecoderVae(f"small-decoder:{kwargs.get('torch_dtype')}")

            def load_state_dict(self, state_dict, strict=True):
                return self.inner.load_state_dict(state_dict, strict=strict)

            def register_to_config(self, **kwargs):
                return self.inner.register_to_config(**kwargs)

            def to(self, dtype=None, device=None):
                return self.inner.to(dtype=dtype, device=device)

    class _FakeDecoder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            record["tokenizer_calls"].append({"model_id": model_id, "kwargs": kwargs})
            return f"tokenizer:{model_id}"

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            record["config_calls"].append({"model_id": model_id, "kwargs": kwargs})
            return {"config_id": model_id}

    fake_diffusers = types.SimpleNamespace(
        Flux2KleinPipeline=_FakeFluxPipelineClass,
        AutoencoderKLFlux2=_FakeAutoencoderKLFlux2,
        Flux2Transformer2DModel=type("Flux2Transformer2DModel", (), {}),
    )
    fake_transformers = types.SimpleNamespace(
        Qwen3ForCausalLM=_FakeTextEncoder,
        AutoTokenizer=_FakeAutoTokenizer,
        AutoConfig=_FakeAutoConfig,
    )
    fake_optimum_quanto = types.SimpleNamespace(
        requantize=lambda model, state_dict, quantization_map: record["requantize_calls"].append(
            {"model": model, "state_dict": state_dict, "quantization_map": quantization_map}
        )
    )
    fake_accelerate = types.SimpleNamespace(init_empty_weights=_fake_init_empty_weights)
    fake_quantized_flux2 = types.SimpleNamespace(
        QuantizedFlux2Transformer2DModel=types.SimpleNamespace(
            from_pretrained=lambda model_path: record["quantized_model_calls"].append(model_path) or _FakeWrappedTransformer()
        )
    )
    fake_sdnq_loader_module = types.SimpleNamespace(
        load_sdnq_model=lambda model_path, model_cls=None, dtype=None, device="cpu": (
            record["sdnq_model_calls"].append(
                {"model_path": model_path, "model_cls": model_cls, "dtype": dtype, "device": device}
            )
            or (
                "sdnq-transformer"
                if model_path.endswith("transformer")
                else "sdnq-text-encoder"
            )
        ),
        load_files=lambda files, key_mapping=None, device="cpu", method="safetensors": {"dummy": 1},
        post_process_model=lambda model: model,
        apply_sdnq_options_to_model=lambda model, dtype=None, dequantize_fp32=None, use_quantized_matmul=True: (
            f"optimized:{model}" if use_quantized_matmul else model
        ),
    )
    fake_autoencoder_module = types.SimpleNamespace(Decoder=_FakeDecoder)
    return {
        "diffusers": fake_diffusers,
        "diffusers.models.autoencoders.autoencoder_kl_flux2": fake_autoencoder_module,
        "transformers": fake_transformers,
        "optimum.quanto": fake_optimum_quanto,
        "accelerate": fake_accelerate,
        "sdnq.loader": fake_sdnq_loader_module,
        "sdnq.quantizer": types.SimpleNamespace(
            get_quant_args_from_config=lambda config: {},
            sdnq_post_load_quant=lambda model, **kwargs: model,
        ),
        "src.image.quantized_flux2": fake_quantized_flux2,
    }


class _FakeSmashConfig:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device
        self.saving_disabled = False

    def disable_saving(self):
        self.saving_disabled = True


def _build_fake_pruna_module(record):
    def _smash(model, smash_config):
        record["smash_calls"].append({"model": model, "smash_config": smash_config})
        smashed = _FakeLoadedPipe()
        smashed.transformer = "pruna-fora-transformer"
        return smashed

    return types.SimpleNamespace(
        SmashConfig=_FakeSmashConfig,
        smash=_smash,
    )


class _PipelineManagerForTests(PipelineManager):
    def __init__(self):
        super().__init__(base_dir=".")
        self.base_pipe = _FakeBasePipe()
        self.load_zimage_calls = []

    def load_zimage_pipeline(self, device="mps", use_full_model=False):
        self.load_zimage_calls.append({"device": device, "use_full_model": use_full_model})
        self.current_device = device
        self.current_model = "zimage-int8"
        self.pipe = self.base_pipe
        return self.base_pipe


class PipelineManagerTests(unittest.TestCase):
    def test_should_enable_cpu_offload_is_flux_pose_only(self):
        manager = PipelineManager(base_dir=".")
        self.assertFalse(manager.should_enable_cpu_offload("zimage-int8", True, "cuda"))
        self.assertFalse(manager.should_enable_cpu_offload("flux2-klein-int8", False, "cuda"))
        self.assertFalse(manager.should_enable_cpu_offload("flux2-klein-int8", True, "cpu"))
        self.assertTrue(manager.should_enable_cpu_offload("flux2-klein-int8", True, "cuda"))

    def test_zimage_img2img_pipeline_is_cached_without_manual_to(self):
        manager = _PipelineManagerForTests()
        manager.optimization_profile = "max_speed"

        fake_diffusers = types.SimpleNamespace(
            AutoPipelineForImage2Image=types.SimpleNamespace(
                from_pipe=lambda base_pipe: _FakeImg2ImgPipe(base_pipe)
            ),
            FlowMatchEulerDiscreteScheduler=types.SimpleNamespace(
                from_config=lambda config, use_beta_sigmas=True: {
                    "config": config,
                    "use_beta_sigmas": use_beta_sigmas,
                }
            ),
        )

        with unittest.mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}), \
             unittest.mock.patch("src.core.pipeline_manager.get_device_vram_gb", return_value=8.0):
            first = manager.get_zimage_img2img_pipeline(device="cuda", use_full_model=False)
            second = manager.get_zimage_img2img_pipeline(device="cuda", use_full_model=False)

        self.assertIs(first, second)
        self.assertIs(first.base_pipe, manager.base_pipe)
        self.assertEqual(first.to_calls, 0)
        self.assertEqual(first.enable_attention_slicing_calls, 0)
        self.assertEqual(first.enable_vae_slicing_calls, 0)
        self.assertEqual(first.vae.enable_tiling_calls, 0)

    def test_runtime_memory_policy_caches_fast_flux_success_without_slicing(self):
        manager = PipelineManager(base_dir=".")
        manager.optimization_profile = "max_speed"
        pipe = _MemoryPolicyPipe()

        with unittest.mock.patch("src.core.pipeline_manager.get_device_vram_gb", return_value=8.0):
            policy = manager.apply_runtime_memory_policy(
                pipe,
                model_key="flux2-klein-int8",
                device="cuda",
                width=768,
                height=768,
            )
            manager.cache_runtime_memory_policy(
                "flux2-klein-int8",
                "txt2img",
                768,
                768,
                policy,
            )

        self.assertFalse(policy["attention_slicing"])
        self.assertFalse(policy["vae_slicing"])
        self.assertEqual(pipe.disable_attention_slicing_calls, 1)
        self.assertEqual(pipe.disable_vae_slicing_calls, 1)
        self.assertEqual(
            manager.get_cached_runtime_memory_policy("flux2-klein-int8", "txt2img", 768, 768),
            policy,
        )

    def test_flux2_int8_loader_uses_small_decoder_vae(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
        }

        fake_modules = _build_fake_flux_modules(record)

        with unittest.mock.patch.dict(sys.modules, fake_modules), \
             unittest.mock.patch("src.core.pipeline_manager.snapshot_download", return_value="C:\\fake-model"), \
             unittest.mock.patch("src.core.pipeline_manager.load_file", return_value={"weights": 1}), \
             unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="{}")), \
             unittest.mock.patch.object(manager, "_apply_pipeline_memory_policy"), \
             unittest.mock.patch.object(manager, "compile_pipeline_components", return_value=False):
            pipe = manager.load_flux2_klein_pipeline(device="cpu")

        self.assertIsInstance(pipe, _FakeLoadedPipe)
        self.assertEqual(record["vae_calls"][0]["repo_id"], "black-forest-labs/FLUX.2-small-decoder")
        self.assertEqual(record["pipeline_calls"][0]["model_id"], "black-forest-labs/FLUX.2-klein-4B")
        self.assertEqual(record["pipeline_calls"][0]["kwargs"]["vae"].label, "small-decoder:torch.float32")
        self.assertEqual(record["pipeline_calls"][0]["kwargs"]["transformer"], None)
        self.assertEqual(pipe.transformer, "wrapped-transformer")
        self.assertIsInstance(pipe.text_encoder, _FakeTextEncoder)
        self.assertEqual(pipe.tokenizer, "tokenizer:C:\\fake-model/tokenizer")

    def test_flux2_sdnq_loader_uses_small_decoder_vae_for_4b(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
        }

        fake_modules = _build_fake_flux_modules(record)

        with unittest.mock.patch.dict(
            sys.modules,
            {
                **fake_modules,
                "sdnq.common": types.SimpleNamespace(use_torch_compile=True),
            },
        ), \
             unittest.mock.patch("src.core.pipeline_manager.snapshot_download", return_value="C:\\fake-sdnq-model"), \
             unittest.mock.patch.object(manager, "_apply_pipeline_memory_policy"):
            pipe = manager.load_flux2_klein_sdnq_pipeline(device="cpu")

        self.assertIsInstance(pipe, _FakeLoadedPipe)
        self.assertEqual(record["vae_calls"][0]["repo_id"], "black-forest-labs/FLUX.2-small-decoder")
        self.assertEqual(
            record["pipeline_calls"][0]["kwargs"]["vae"].label,
            "small-decoder:torch.float32",
        )
        self.assertEqual(pipe.transformer, "sdnq-transformer")
        self.assertEqual(pipe.text_encoder, "sdnq-text-encoder")
        self.assertEqual(record["sdnq_model_calls"][0]["model_path"], "C:\\fake-sdnq-model\\transformer")
        self.assertEqual(record["sdnq_model_calls"][1]["model_path"], "C:\\fake-sdnq-model\\text_encoder")

    def test_flux2_small_decoder_compatibility_builder_uses_same_hf_repo_weights(self):
        manager = PipelineManager(base_dir=".")
        fake_modules = _build_fake_flux_modules({})
        downloads = []

        def _fake_download(repo_id, filename):
            downloads.append((repo_id, filename))
            return f"C:\\hf\\{filename}"

        with unittest.mock.patch.dict(sys.modules, fake_modules), \
             unittest.mock.patch("src.core.pipeline_manager.hf_hub_download", side_effect=_fake_download), \
             unittest.mock.patch("src.core.pipeline_manager.load_file", return_value={"weights": 1}), \
             unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data='{"in_channels":3,"out_channels":3,"down_block_types":["DownEncoderBlock2D"],"up_block_types":["UpDecoderBlock2D"],"block_out_channels":[64],"decoder_block_out_channels":[96,192,384,384],"layers_per_block":2,"act_fn":"silu","latent_channels":32,"norm_num_groups":32,"sample_size":1024,"force_upcast":true,"use_quant_conv":true,"use_post_quant_conv":true,"mid_block_add_attention":true,"batch_norm_eps":0.0001,"batch_norm_momentum":0.1,"patch_size":[2,2]}')):
            vae = manager._build_asymmetric_flux2_small_decoder_vae(dtype="float32")

        self.assertIsInstance(vae, _FakeSmallDecoderVae)
        self.assertEqual(
            downloads,
            [
                ("black-forest-labs/FLUX.2-small-decoder", "config.json"),
                ("black-forest-labs/FLUX.2-small-decoder", "diffusion_pytorch_model.safetensors"),
            ],
        )
        self.assertEqual(vae.loaded_state_dict, ({"weights": 1}, True))
        self.assertEqual(
            vae.registered_config,
            {"decoder_block_out_channels": (96, 192, 384, 384)},
        )

    def test_flux2_sdnq_loader_allows_missing_guidance_weights(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
        }

        class _StrictFlux2Transformer2DModel:
            def __init__(self, *args, **kwargs):
                self.config = types.SimpleNamespace(guidance_embeds=False)
                self.load_state_dict_calls = []

            def load_state_dict(self, state_dict, strict=True, assign=False):
                self.load_state_dict_calls.append({"strict": strict, "assign": assign})
                if strict:
                    raise RuntimeError(
                        "Missing key(s) in state_dict: "
                        "\"time_guidance_embed.guidance_embedder.linear_1.weight\", "
                        "\"time_guidance_embed.guidance_embedder.linear_2.weight\""
                    )
                return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def parameters(self):
                return []

        fake_modules = _build_fake_flux_modules(record)
        fake_modules["diffusers"].Flux2Transformer2DModel = _StrictFlux2Transformer2DModel
        def _fake_load_sdnq_model(model_path, model_cls=None, dtype=None, device="cpu"):
            record["sdnq_model_calls"].append(
                {"model_path": model_path, "model_cls": model_cls, "dtype": dtype, "device": device}
            )
            if model_path.endswith("transformer"):
                model = model_cls()
                model.load_state_dict({"dummy": 1}, assign=True)
                return model
            return "sdnq-text-encoder"

        fake_modules["sdnq.loader"] = types.SimpleNamespace(
            load_sdnq_model=_fake_load_sdnq_model,
            load_files=lambda files, key_mapping=None, device="cpu", method="safetensors": {"dummy": 1},
            post_process_model=lambda model: model,
            apply_sdnq_options_to_model=lambda model, dtype=None, dequantize_fp32=None, use_quantized_matmul=True: model,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {
                **fake_modules,
                "sdnq.common": types.SimpleNamespace(use_torch_compile=False),
            },
        ), \
             unittest.mock.patch("src.core.pipeline_manager.snapshot_download", return_value="C:\\fake-sdnq-model"), \
             unittest.mock.patch.object(manager, "_apply_pipeline_memory_policy"):
            pipe = manager.load_flux2_klein_sdnq_pipeline(device="cpu")

        self.assertIsInstance(pipe, _FakeLoadedPipe)
        self.assertEqual(record["sdnq_model_calls"][0]["model_path"], "C:\\fake-sdnq-model\\transformer")
        self.assertEqual(pipe.transformer.__class__.__name__, "_StrictFlux2Transformer2DModel")
        self.assertEqual(pipe.transformer.load_state_dict_calls[0]["strict"], False)
        self.assertEqual(pipe.transformer.load_state_dict_calls[0]["assign"], True)

    def test_flux2_sdnq_text_encoder_falls_back_from_qwen3_loader_bug(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
            "load_files_calls": [],
        }

        class _FallbackTextEncoder(_FakeTextEncoder):
            def __init__(self, config):
                super().__init__(config)
                self.loaded_state_dict = None

            def load_state_dict(self, state_dict, strict=True, assign=False):
                self.loaded_state_dict = {
                    "state_dict": dict(state_dict),
                    "strict": strict,
                    "assign": assign,
                }
                return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

        fake_modules = _build_fake_flux_modules(record)
        fake_modules["transformers"].Qwen3ForCausalLM = _FallbackTextEncoder

        def _broken_load_sdnq_model(model_path, model_cls=None, dtype=None, device="cpu"):
            record["sdnq_model_calls"].append(
                {"model_path": model_path, "model_cls": model_cls, "dtype": dtype, "device": device}
            )
            raise UnboundLocalError("cannot access local variable 'transformers' where it is not associated with a value")

        fake_modules["sdnq.loader"] = types.SimpleNamespace(
            load_sdnq_model=_broken_load_sdnq_model,
            apply_sdnq_options_to_model=lambda model, dtype=None, dequantize_fp32=None, use_quantized_matmul=None: model,
            load_files=lambda files, key_mapping=None, device="cpu", method="safetensors": (
                record["load_files_calls"].append(
                    {"files": list(files), "key_mapping": key_mapping, "device": device, "method": method}
                )
                or {"model.embed_tokens.weight": "embed"}
            ),
            post_process_model=lambda model: model,
        )
        fake_modules["sdnq.quantizer"] = types.SimpleNamespace(
            get_quant_args_from_config=lambda config: {"quant_config": config},
            sdnq_post_load_quant=lambda model, **kwargs: model,
        )

        with unittest.mock.patch.dict(sys.modules, fake_modules), \
             unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="{}")), \
             unittest.mock.patch("os.listdir", return_value=["model.safetensors", "quantization_config.json"]):
            text_encoder = manager._load_flux2_klein_sdnq_text_encoder(
                "C:\\fake-sdnq-model",
                dtype="float32",
                device="cpu",
            )

        self.assertIsInstance(text_encoder, _FallbackTextEncoder)
        self.assertEqual(record["sdnq_model_calls"][0]["model_path"], "C:\\fake-sdnq-model\\text_encoder")
        self.assertEqual(
            record["load_files_calls"][0]["files"],
            ["C:\\fake-sdnq-model\\text_encoder\\model.safetensors"],
        )
        self.assertEqual(text_encoder.loaded_state_dict["state_dict"]["lm_head.weight"], "embed")
        self.assertTrue(text_encoder.eval_called)

    def test_flux2_chat_template_wrapper_flattens_text_only_message_lists(self):
        manager = PipelineManager(base_dir=".")
        tokenizer = _FakeChatTemplateTokenizer()
        wrapped = manager._wrap_flux2_chat_template_tokenizer(tokenizer)

        result = wrapped.apply_chat_template(
            [
                [
                    {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                ]
            ],
            tokenize=True,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(wrapped.calls[0]["messages"][0][0]["content"], "sys")
        self.assertEqual(wrapped.calls[0]["messages"][0][1]["content"], "hello")
        self.assertTrue(wrapped._ufig_flux2_chat_template_wrapped)

    def test_flux2_small_decoder_failure_uses_local_compatibility_builder(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
        }

        fake_modules = _build_fake_flux_modules(record, small_decoder_side_effect=RuntimeError("decoder load failed"))
        manual_vae = _FakeSmallDecoderVae("manual-compatibility")

        with unittest.mock.patch.dict(sys.modules, fake_modules), \
             unittest.mock.patch("src.core.pipeline_manager.snapshot_download", return_value="C:\\fake-model"), \
             unittest.mock.patch("src.core.pipeline_manager.load_file", return_value={"weights": 1}), \
             unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="{}")), \
             unittest.mock.patch.object(manager, "_build_asymmetric_flux2_small_decoder_vae", return_value=manual_vae), \
             unittest.mock.patch.object(manager, "_apply_pipeline_memory_policy"), \
             unittest.mock.patch.object(manager, "compile_pipeline_components", return_value=False):
            pipe = manager.load_flux2_klein_pipeline(device="cpu")

        self.assertIsInstance(pipe, _FakeLoadedPipe)
        self.assertIs(record["pipeline_calls"][0]["kwargs"]["vae"], manual_vae)

    def test_flux2_small_decoder_unsupported_diffusers_uses_local_builder(self):
        manager = PipelineManager(base_dir=".")
        record = {
            "pipeline_calls": [],
            "vae_calls": [],
            "tokenizer_calls": [],
            "config_calls": [],
            "requantize_calls": [],
            "quantized_model_calls": [],
            "sdnq_model_calls": [],
        }

        fake_modules = _build_fake_flux_modules(record, unsupported_diffusers=True)
        manual_vae = _FakeSmallDecoderVae("manual-unsupported")

        with unittest.mock.patch.dict(sys.modules, fake_modules), \
             unittest.mock.patch("src.core.pipeline_manager.snapshot_download", return_value="C:\\fake-model"), \
             unittest.mock.patch("src.core.pipeline_manager.load_file", return_value={"weights": 1}), \
             unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="{}")), \
             unittest.mock.patch.object(manager, "_build_asymmetric_flux2_small_decoder_vae", return_value=manual_vae), \
             unittest.mock.patch.object(manager, "_apply_pipeline_memory_policy"), \
             unittest.mock.patch.object(manager, "compile_pipeline_components", return_value=False):
            pipe = manager.load_flux2_klein_pipeline(device="cpu")

        self.assertIsInstance(pipe, _FakeLoadedPipe)
        self.assertIs(record["pipeline_calls"][0]["kwargs"]["vae"], manual_vae)

    def test_flux_sdnq_pruna_smash_config_uses_guidance_free_defaults(self):
        manager = PipelineManager(base_dir=".")
        record = {"smash_calls": []}

        with unittest.mock.patch.dict(sys.modules, {"pruna": _build_fake_pruna_module(record)}):
            smash_config = manager._build_flux_sdnq_pruna_smash_config(device="cuda")

        self.assertEqual(smash_config.device, "cuda")
        self.assertTrue(smash_config.saving_disabled)
        self.assertEqual(
            smash_config.config,
            {
                "fora": {
                    "fora_interval": 2,
                    "fora_start_step": 2,
                    "fora_backbone_calls_per_step": 1,
                }
            },
        )

    def test_prepare_flux_sdnq_optional_accelerators_skips_low_step_runs(self):
        manager = PipelineManager(base_dir=".")
        manager.current_model = "flux2-klein-sdnq"
        pipe = _FakeLoadedPipe()

        accelerated_pipe, status = manager.prepare_flux_sdnq_optional_accelerators(
            pipe,
            device="cuda",
            steps=6,
            enable_optional_accelerators=True,
        )

        self.assertIs(accelerated_pipe, pipe)
        self.assertFalse(status["enabled"])
        self.assertTrue(status["requested"])
        self.assertTrue(status["small_decoder"])
        self.assertEqual(status["skip_reason"], "requires at least 8 inference steps")

    def test_prepare_flux_sdnq_optional_accelerators_enables_pruna_for_eligible_runs(self):
        manager = PipelineManager(base_dir=".")
        manager.current_model = "flux2-klein-sdnq"
        pipe = _FakeLoadedPipe()
        record = {"smash_calls": []}

        with unittest.mock.patch.dict(sys.modules, {"pruna": _build_fake_pruna_module(record)}):
            accelerated_pipe, status = manager.prepare_flux_sdnq_optional_accelerators(
                pipe,
                device="cuda",
                steps=8,
                enable_optional_accelerators=True,
            )

        self.assertIsNot(accelerated_pipe, pipe)
        self.assertEqual(accelerated_pipe.transformer, "pruna-fora-transformer")
        self.assertTrue(status["enabled"])
        self.assertTrue(status["pruna_fora"])
        self.assertEqual(status["backend"], "pruna-fora")
        self.assertEqual(record["smash_calls"][0]["smash_config"].config["fora"]["fora_backbone_calls_per_step"], 1)

    def test_prepare_flux_sdnq_optional_accelerators_gates_modifier_workflows(self):
        manager = PipelineManager(base_dir=".")
        manager.current_model = "flux2-klein-sdnq"
        pipe = _FakeLoadedPipe()

        scenarios = [
            ({"mode": "batch"}, "disabled for batch workflow"),
            ({"mode": "video"}, "disabled for video workflow"),
            ({"has_lora": True}, "disabled when LoRA adapters are active"),
            ({"has_pulid": True}, "disabled when PuLID patching is active"),
            ({"has_faceswap": True}, "disabled when FaceSwap post-processing is active"),
            ({"has_pose_control": True}, "disabled for pose/controlnet workflows"),
            ({"has_cpu_offload": True}, "disabled when CPU offload would be enabled"),
        ]

        for extra_kwargs, expected_reason in scenarios:
            with self.subTest(expected_reason=expected_reason):
                accelerated_pipe, status = manager.prepare_flux_sdnq_optional_accelerators(
                    pipe,
                    device="cuda",
                    steps=8,
                    enable_optional_accelerators=True,
                    **extra_kwargs,
                )
                self.assertIs(accelerated_pipe, pipe)
                self.assertFalse(status["enabled"])
                self.assertEqual(status["skip_reason"], expected_reason)

    def test_get_model_repos_for_choice_includes_small_decoder_for_flux_4b_only(self):
        manager = PipelineManager(base_dir=".")

        self.assertEqual(
            manager.get_model_repos_for_choice("Z-Image Turbo (Int8 - 8GB Safe)"),
            ["Disty0/Z-Image-Turbo-SDNQ-int8"],
        )
        self.assertEqual(
            manager.get_model_repos_for_choice("FLUX.2-klein-4B (Int8)"),
            ["aydin99/FLUX.2-klein-4B-int8", "black-forest-labs/FLUX.2-small-decoder"],
        )
        self.assertEqual(
            manager.get_model_repos_for_choice("FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"),
            [
                "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
                "black-forest-labs/FLUX.2-klein-4B",
                "black-forest-labs/FLUX.2-small-decoder",
            ],
        )
        self.assertNotIn(
            "black-forest-labs/FLUX.2-small-decoder",
            manager.get_model_repos_for_choice("Z-Image Turbo (Int8 - 8GB Safe)"),
        )

    def test_zimage_int8_choice_keeps_separate_pipeline(self):
        manager = _PipelineManagerForTests()

        pipe = manager.load_pipeline("Z-Image Turbo (Int8 - 8GB Safe)", device="cuda")

        self.assertIs(pipe, manager.base_pipe)
        self.assertEqual(manager.current_model, "zimage-int8")
        self.assertEqual(manager.load_zimage_calls[0]["use_full_model"], False)

    def test_unsupported_model_choice_falls_back_to_zimage_int8_route(self):
        manager = PipelineManager(base_dir=".")
        self.assertEqual(
            manager.get_model_repos_for_choice("FLUX.2-klein-9B (4bit SDNQ - Higher Quality)"),
            ["Disty0/Z-Image-Turbo-SDNQ-int8"],
        )
        self.assertEqual(
            manager.get_model_repos_for_choice("FLUX.2-klein-4B (NVFP4 - Experimental)"),
            ["Disty0/Z-Image-Turbo-SDNQ-int8"],
        )
        self.assertEqual(
            manager.get_model_repos_for_choice("Z-Image Turbo (Int8 - 8GB Safe)"),
            ["Disty0/Z-Image-Turbo-SDNQ-int8"],
        )


if __name__ == "__main__":
    unittest.main()
