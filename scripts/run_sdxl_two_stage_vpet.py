#!/usr/bin/env python3
"""Run the SDXL V-Pet pipeline as prepare -> pose normalize -> refine."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGE1_PROMPT = (
    "{name}, same character as input image, preserve species anatomy and signature traits, preserve silhouette family, "
    "convert to vpet_style, vpet_left_pose, partial left-facing, three-quarter view, canonical v-pet stance, "
    "digimon, full body, sprite, pixel art, clean 1px dark outline, flat cel shading, simplified forms, limited color palette"
)
STAGE1_NEGATIVE = (
    "different character, changed species, changed face, changed silhouette, changed anatomy, right-facing, "
    "front-facing, profile-only, photorealistic, painterly, texture heavy, blurry, noisy, text, watermark"
)
STAGE2_PROMPT = (
    "{name}, same character as input image, preserve vpet_left_pose, preserve species anatomy and signature traits, "
    "refine into vpet_style, partial left-facing, three-quarter view, digimon, full body, sprite, pixel art, "
    "clean 1px dark outline, flat cel shading, simplified shapes, reduced texture noise, limited color palette"
)
STAGE2_NEGATIVE = (
    "different character, changed species, changed face, changed silhouette, changed anatomy, right-facing, "
    "front-facing, profile-only, photorealistic, painterly, texture heavy, blurry, noisy, text, watermark"
)


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL two-stage V-Pet conversion pipeline.")
    parser.add_argument("--input-dir", default="working", help="Source input directory with raw official images.")
    parser.add_argument("--prepared-dir", default="outputs/sdxl_two_stage/prepared", help="Prepared Stage 1 input directory.")
    parser.add_argument("--stage1-dir", default="outputs/sdxl_two_stage/stage1_pose", help="Stage 1 output directory.")
    parser.add_argument("--output-dir", default="outputs/sdxl_two_stage/final", help="Final Stage 2 output directory.")
    parser.add_argument(
        "--stage1-workflow",
        default="workflows/vpet_sdxl_lora_pose_stage1_api.json",
        help="Workflow JSON for pose-normalization stage.",
    )
    parser.add_argument(
        "--stage2-workflow",
        default="workflows/vpet_sdxl_lora_refine_stage2_api.json",
        help="Workflow JSON for refinement stage.",
    )
    parser.add_argument("--checkpoint", required=True, help="Installed SDXL checkpoint filename in ComfyUI.")
    parser.add_argument("--lora", required=True, help="Installed V-Pet SDXL LoRA filename in ComfyUI.")
    parser.add_argument("--comfy-input-dir", required=True, help="ComfyUI input directory path.")
    parser.add_argument("--comfy-output-dir", required=True, help="ComfyUI output directory path.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI API base URL.")
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all).")
    parser.add_argument("--prompt-timeout", type=int, default=1800, help="Max wait time per ComfyUI job.")
    parser.add_argument("--stage1-steps", type=int, default=22, help="Override Stage 1 sampler steps.")
    parser.add_argument("--stage1-cfg", type=float, default=4.8, help="Override Stage 1 CFG.")
    parser.add_argument("--stage1-denoise", type=float, default=0.46, help="Override Stage 1 denoise.")
    parser.add_argument("--stage1-lora-strength", type=float, default=0.72, help="Override Stage 1 LoRA strength.")
    parser.add_argument("--stage2-steps", type=int, default=18, help="Override Stage 2 sampler steps.")
    parser.add_argument("--stage2-cfg", type=float, default=4.6, help="Override Stage 2 CFG.")
    parser.add_argument("--stage2-denoise", type=float, default=0.24, help="Override Stage 2 denoise.")
    parser.add_argument("--stage2-lora-strength", type=float, default=0.86, help="Override Stage 2 LoRA strength.")
    parser.add_argument(
        "--disable-auto-flip-right-facing",
        action="store_true",
        help="Disable the default heuristic that flips obviously right-facing source images during prepare step.",
    )
    parser.add_argument(
        "--flip-score-threshold",
        type=float,
        default=0.06,
        help="Confidence threshold used by the prepare-step facing heuristic.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip ComfyUI model existence checks in both stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    py = sys.executable

    prepare_cmd = [
        py,
        "scripts/prepare_working_inputs.py",
        "--input",
        args.input_dir,
        "--output",
        args.prepared_dir,
    ]
    if args.limit > 0:
        prepare_cmd.extend(["--limit", str(args.limit)])
    if not args.disable_auto_flip_right_facing:
        prepare_cmd.append("--auto-flip-right-facing")
        prepare_cmd.extend(["--flip-score-threshold", str(args.flip_score_threshold)])
    run(prepare_cmd, cwd=repo_dir)

    stage1_cmd = [
        py,
        "scripts/run_comfy_batch_img2img.py",
        "--workflow",
        args.stage1_workflow,
        "--input-dir",
        args.prepared_dir,
        "--output-dir",
        args.stage1_dir,
        "--checkpoint",
        args.checkpoint,
        "--lora",
        args.lora,
        "--lora-strength",
        str(args.stage1_lora_strength),
        "--steps",
        str(args.stage1_steps),
        "--cfg",
        str(args.stage1_cfg),
        "--denoise",
        str(args.stage1_denoise),
        "--prompt-template",
        STAGE1_PROMPT,
        "--negative-prompt",
        STAGE1_NEGATIVE,
        "--comfy-url",
        args.comfy_url,
        "--comfy-input-dir",
        args.comfy_input_dir,
        "--comfy-output-dir",
        args.comfy_output_dir,
        "--prompt-timeout",
        str(args.prompt_timeout),
    ]
    if args.limit > 0:
        stage1_cmd.extend(["--limit", str(args.limit)])
    if args.skip_model_check:
        stage1_cmd.append("--skip-model-check")
    run(stage1_cmd, cwd=repo_dir)

    stage2_cmd = [
        py,
        "scripts/run_comfy_batch_img2img.py",
        "--workflow",
        args.stage2_workflow,
        "--input-dir",
        args.stage1_dir,
        "--output-dir",
        args.output_dir,
        "--checkpoint",
        args.checkpoint,
        "--lora",
        args.lora,
        "--lora-strength",
        str(args.stage2_lora_strength),
        "--steps",
        str(args.stage2_steps),
        "--cfg",
        str(args.stage2_cfg),
        "--denoise",
        str(args.stage2_denoise),
        "--prompt-template",
        STAGE2_PROMPT,
        "--negative-prompt",
        STAGE2_NEGATIVE,
        "--postprocess-sprite",
        "--comfy-url",
        args.comfy_url,
        "--comfy-input-dir",
        args.comfy_input_dir,
        "--comfy-output-dir",
        args.comfy_output_dir,
        "--prompt-timeout",
        str(args.prompt_timeout),
    ]
    if args.limit > 0:
        stage2_cmd.extend(["--limit", str(args.limit)])
    if args.skip_model_check:
        stage2_cmd.append("--skip-model-check")
    run(stage2_cmd, cwd=repo_dir)


if __name__ == "__main__":
    main()
