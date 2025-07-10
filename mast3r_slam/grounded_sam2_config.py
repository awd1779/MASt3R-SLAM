"""Grounded-SAM2 Model Configuration for MAST3R-SLAM Integration"""

import torch
from typing import Dict, Tuple, Optional

# Model configurations with performance characteristics
SAM2_MODELS = {
    "hiera_tiny": {
        "checkpoint": "sam2_hiera_tiny.pt",
        "config": "sam2_hiera_t.yaml",
        "params": 39e6,
        "fps_estimate": 50,  # on V100
        "vram_gb": 4,
        "accuracy": "good",
        "use_case": "real-time applications"
    },
    "hiera_small": {
        "checkpoint": "sam2_hiera_small.pt", 
        "config": "sam2_hiera_s.yaml",
        "params": 46e6,
        "fps_estimate": 35,
        "vram_gb": 5,
        "accuracy": "better",
        "use_case": "balanced performance"
    },
    "hiera_b+": {
        "checkpoint": "sam2_hiera_base_plus.pt",
        "config": "sam2_hiera_b+.yaml", 
        "params": 80.8e6,
        "fps_estimate": 20,
        "vram_gb": 6,
        "accuracy": "high",
        "use_case": "research and high-quality reconstruction"
    },
    "hiera_large": {
        "checkpoint": "sam2_hiera_large.pt",
        "config": "sam2_hiera_l.yaml",
        "params": 224.4e6,
        "fps_estimate": 10,
        "vram_gb": 10,
        "accuracy": "best",
        "use_case": "offline processing"
    }
}

GROUNDING_MODELS = {
    "grounding_dino_swin-t": {
        "checkpoint": "groundingdino_swint_ogc.pth",
        "config": "GroundingDINO_SwinT_OGC.py",
        "params": 23e6,
        "fps_estimate": 40,
        "vram_gb": 3,
        "accuracy": "good for simple scenes"
    },
    "grounding_dino_swin-b": {
        "checkpoint": "groundingdino_swinb_cogcoor.pth",
        "config": "GroundingDINO_SwinB_cfg.py",
        "params": 88e6,
        "fps_estimate": 25,
        "vram_gb": 4,
        "accuracy": "excellent for complex scenes"
    },
    "grounding_dino_swin-l": {
        "checkpoint": "groundingdino_swinl_cogcoor.pth",
        "config": "GroundingDINO_SwinL_cfg.py",
        "params": 197e6,
        "fps_estimate": 15,
        "vram_gb": 6,
        "accuracy": "best, handles rare objects"
    }
}

# Recommended configurations for different scenarios
RECOMMENDED_CONFIGS = {
    "realtime": {
        "sam2": "hiera_tiny",
        "grounding": "grounding_dino_swin-t",
        "expected_fps": 30,
        "total_vram_gb": 7,
        "notes": "For live camera SLAM with semantic segmentation"
    },
    "balanced": {
        "sam2": "hiera_small",
        "grounding": "grounding_dino_swin-b",
        "expected_fps": 20,
        "total_vram_gb": 9,
        "notes": "Good balance for most research applications"
    },
    "quality": {
        "sam2": "hiera_b+",
        "grounding": "grounding_dino_swin-b",
        "expected_fps": 15,
        "total_vram_gb": 10,
        "notes": "High quality for paper results and evaluations"
    },
    "offline": {
        "sam2": "hiera_large",
        "grounding": "grounding_dino_swin-l",
        "expected_fps": 8,
        "total_vram_gb": 16,
        "notes": "Best quality for offline processing"
    }
}


class GroundedSAM2ModelSelector:
    """Helper class to select appropriate models based on requirements"""
    
    def __init__(self, target_fps: Optional[float] = None, 
                 max_vram_gb: Optional[float] = None,
                 quality_priority: str = "balanced"):
        """
        Args:
            target_fps: Minimum required FPS (None for no constraint)
            max_vram_gb: Maximum VRAM available (None for no constraint)
            quality_priority: "speed", "balanced", or "quality"
        """
        self.target_fps = target_fps
        self.max_vram_gb = max_vram_gb
        self.quality_priority = quality_priority
        
    def select_models(self) -> Tuple[str, str]:
        """Select SAM2 and Grounding DINO models based on constraints"""
        
        # Start with recommended config
        if self.quality_priority == "speed":
            base_config = RECOMMENDED_CONFIGS["realtime"]
        elif self.quality_priority == "quality":
            base_config = RECOMMENDED_CONFIGS["quality"]
        else:
            base_config = RECOMMENDED_CONFIGS["balanced"]
            
        sam2_model = base_config["sam2"]
        grounding_model = base_config["grounding"]
        
        # Check constraints
        if self.target_fps is not None:
            # Find fastest combination that meets FPS requirement
            for config_name, config in RECOMMENDED_CONFIGS.items():
                if config["expected_fps"] >= self.target_fps:
                    sam2_model = config["sam2"]
                    grounding_model = config["grounding"]
                    break
                    
        if self.max_vram_gb is not None:
            # Ensure we don't exceed VRAM limit
            current_vram = (SAM2_MODELS[sam2_model]["vram_gb"] + 
                          GROUNDING_MODELS[grounding_model]["vram_gb"])
            
            if current_vram > self.max_vram_gb:
                # Downgrade models
                if sam2_model == "hiera_large":
                    sam2_model = "hiera_b+"
                elif sam2_model == "hiera_b+":
                    sam2_model = "hiera_small"
                elif sam2_model == "hiera_small":
                    sam2_model = "hiera_tiny"
                    
                if grounding_model == "grounding_dino_swin-l":
                    grounding_model = "grounding_dino_swin-b"
                elif grounding_model == "grounding_dino_swin-b":
                    grounding_model = "grounding_dino_swin-t"
                    
        return sam2_model, grounding_model
    
    def get_model_info(self, sam2_model: str, grounding_model: str) -> Dict:
        """Get detailed information about selected models"""
        sam2_info = SAM2_MODELS[sam2_model]
        grounding_info = GROUNDING_MODELS[grounding_model]
        
        total_params = sam2_info["params"] + grounding_info["params"]
        total_vram = sam2_info["vram_gb"] + grounding_info["vram_gb"]
        min_fps = min(sam2_info["fps_estimate"], grounding_info["fps_estimate"])
        
        return {
            "sam2": {
                "model": sam2_model,
                "checkpoint": sam2_info["checkpoint"],
                "params": f"{sam2_info['params']/1e6:.1f}M",
                "vram": f"{sam2_info['vram_gb']}GB",
                "fps": sam2_info["fps_estimate"]
            },
            "grounding": {
                "model": grounding_model,
                "checkpoint": grounding_info["checkpoint"],
                "params": f"{grounding_info['params']/1e6:.1f}M",
                "vram": f"{grounding_info['vram_gb']}GB",
                "fps": grounding_info["fps_estimate"]
            },
            "combined": {
                "total_params": f"{total_params/1e6:.1f}M",
                "total_vram": f"{total_vram}GB",
                "expected_fps": min_fps,
                "bottleneck": "sam2" if sam2_info["fps_estimate"] < grounding_info["fps_estimate"] else "grounding"
            }
        }


def benchmark_model_selection():
    """Benchmark different model configurations"""
    print("Grounded-SAM2 Model Configuration Analysis\n")
    print("=" * 60)
    
    scenarios = [
        ("Live Camera SLAM", {"target_fps": 25, "quality_priority": "speed"}),
        ("Research Quality", {"quality_priority": "quality"}),
        ("Limited GPU (8GB)", {"max_vram_gb": 8, "quality_priority": "balanced"}),
        ("Best Possible", {"quality_priority": "quality", "max_vram_gb": 24})
    ]
    
    for scenario_name, kwargs in scenarios:
        selector = GroundedSAM2ModelSelector(**kwargs)
        sam2, grounding = selector.select_models()
        info = selector.get_model_info(sam2, grounding)
        
        print(f"\n{scenario_name}:")
        print(f"  SAM2: {info['sam2']['model']} ({info['sam2']['params']}, {info['sam2']['fps']} FPS)")
        print(f"  Grounding: {info['grounding']['model']} ({info['grounding']['params']}, {info['grounding']['fps']} FPS)")
        print(f"  Combined: {info['combined']['total_vram']} VRAM, ~{info['combined']['expected_fps']} FPS")
        print(f"  Bottleneck: {info['combined']['bottleneck']}")


if __name__ == "__main__":
    benchmark_model_selection()