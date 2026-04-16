"""UI state dataclass for type-safe state management."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class UIState:
    """Dataclass for UI state management."""
    model_choice: str
    prompt: str = ""
    negative_prompt: str = ""
    preset_choice: str = "None"
    resolution_preset: str = "~768px"
    batch_input_folder: str = ""
    batch_output_folder: str = ""
    batch_resolution_preset: str = "~768px"
    downscale_factor: str = "1x"
    img2img_strength: float = 0.6
    height: int = 512
    width: int = 512
    steps: int = 4
    seed: int = -1
    guidance_scale: float = 0.0
    enable_klein_anatomy_fix: bool = False
    device: str = "cuda"
    lora_file: Optional[str] = None
    lora_strength: float = 1.0
    enable_multi_character: bool = False
    character_input_folder: str = ""
    character_description: str = ""
    enable_faceswap: bool = False
    faceswap_target_index: int = 0
    optimization_profile: str = "balanced"
    enable_windows_compile_probe: bool = False
    enable_cuda_graphs: bool = False
    enable_optional_accelerators: bool = False
    enable_pose_preservation: bool = False
    pose_detector_type: str = "dwpose"
    pose_mode: str = "Body + Face"
    controlnet_strength: float = 0.7
    show_pose_skeleton: bool = False
    enable_gender_preservation: bool = True
    gender_strength: float = 1.0
    enable_prompt_upsampling: bool = False
    video_output_path: str = ""
    preserve_audio: bool = True
    video_resolution_preset: str = "~1024px"
    input_image_names: List[str] = field(default_factory=list)
    faceswap_source_names: List[str] = field(default_factory=list)
    character_reference_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_choice": self.model_choice,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "preset_choice": self.preset_choice,
            "resolution_preset": self.resolution_preset,
            "batch_input_folder": self.batch_input_folder,
            "batch_output_folder": self.batch_output_folder,
            "batch_resolution_preset": self.batch_resolution_preset,
            "downscale_factor": self.downscale_factor,
            "img2img_strength": self.img2img_strength,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "enable_klein_anatomy_fix": self.enable_klein_anatomy_fix,
            "device": self.device,
            "lora_file": self.lora_file,
            "lora_strength": self.lora_strength,
            "enable_multi_character": self.enable_multi_character,
            "character_input_folder": self.character_input_folder,
            "character_description": self.character_description,
            "enable_faceswap": self.enable_faceswap,
            "faceswap_target_index": self.faceswap_target_index,
            "optimization_profile": self.optimization_profile,
            "enable_windows_compile_probe": self.enable_windows_compile_probe,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "enable_optional_accelerators": self.enable_optional_accelerators,
            "enable_pose_preservation": self.enable_pose_preservation,
            "pose_detector_type": self.pose_detector_type,
            "pose_mode": self.pose_mode,
            "controlnet_strength": self.controlnet_strength,
            "show_pose_skeleton": self.show_pose_skeleton,
            "enable_gender_preservation": self.enable_gender_preservation,
            "gender_strength": self.gender_strength,
            "enable_prompt_upsampling": self.enable_prompt_upsampling,
            "video_output_path": self.video_output_path,
            "preserve_audio": self.preserve_audio,
            "video_resolution_preset": self.video_resolution_preset,
            "input_image_names": self.input_image_names,
            "faceswap_source_names": self.faceswap_source_names,
            "character_reference_names": self.character_reference_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UIState":
        """Create from dictionary for JSON deserialization."""
        return cls(
            model_choice=data.get("model_choice", ""),
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            preset_choice=data.get("preset_choice", "None"),
            resolution_preset=data.get("resolution_preset", "~768px"),
            batch_input_folder=data.get("batch_input_folder", ""),
            batch_output_folder=data.get("batch_output_folder", ""),
            batch_resolution_preset=data.get("batch_resolution_preset", "~768px"),
            downscale_factor=data.get("downscale_factor", "1x"),
            img2img_strength=data.get("img2img_strength", 0.6),
            height=data.get("height", 512),
            width=data.get("width", 512),
            steps=data.get("steps", 4),
            seed=data.get("seed", -1),
            guidance_scale=data.get("guidance_scale", 0.0),
            enable_klein_anatomy_fix=data.get("enable_klein_anatomy_fix", False),
            device=data.get("device", "cuda"),
            lora_file=data.get("lora_file", None),
            lora_strength=data.get("lora_strength", 1.0),
            enable_multi_character=data.get("enable_multi_character", False),
            character_input_folder=data.get("character_input_folder", ""),
            character_description=data.get("character_description", ""),
            enable_faceswap=data.get("enable_faceswap", False),
            faceswap_target_index=data.get("faceswap_target_index", 0),
            optimization_profile=data.get("optimization_profile", "balanced"),
            enable_windows_compile_probe=data.get("enable_windows_compile_probe", False),
            enable_cuda_graphs=data.get("enable_cuda_graphs", False),
            enable_optional_accelerators=data.get("enable_optional_accelerators", False),
            enable_pose_preservation=data.get("enable_pose_preservation", False),
            pose_detector_type=data.get("pose_detector_type", "dwpose"),
            pose_mode=data.get("pose_mode", "Body + Face"),
            controlnet_strength=data.get("controlnet_strength", 0.7),
            show_pose_skeleton=data.get("show_pose_skeleton", False),
            enable_gender_preservation=data.get("enable_gender_preservation", True),
            gender_strength=data.get("gender_strength", 1.0),
            enable_prompt_upsampling=data.get("enable_prompt_upsampling", False),
            video_output_path=data.get("video_output_path", ""),
            preserve_audio=data.get("preserve_audio", True),
            video_resolution_preset=data.get("video_resolution_preset", "~1024px"),
            input_image_names=data.get("input_image_names", []),
            faceswap_source_names=data.get("faceswap_source_names", []),
            character_reference_names=data.get("character_reference_names", []),
        )

    @classmethod
    def load_payload(cls, data: Dict[str, Any]) -> "UIState":
        """Create an instance from a persisted payload."""
        return cls.from_dict(data or {})
