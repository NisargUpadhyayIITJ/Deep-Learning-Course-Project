import argparse
import glob
import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

import gradio_dit4sr as base_demo
from utils_data.append_ocr_to_prompts import OCR_SEPARATOR, pil_image_to_base64


args = None
ocr_client = None

OCR_INSTRUCTION = "<|img|><|imgpad|><|endofimg|>Extract the text content from this image."

INTRO = """
## DiT4SR OCR-Prompt Demo

This demo follows the repo's OCR-aware training recipe: it extracts text from the input image with `dots.ocr`
and appends it to the image prompt using the same ` OCR Text: ` format used by the dataset tools.
"""

WORKFLOW = """
1. Generate a visual caption with LLaVA or write your own prompt.
2. Extract OCR text from the input image.
3. Compose the OCR-augmented prompt or use the one-click OCR-enhanced SR button.
4. Compare the bicubic-upscaled input with the DiT4SR result using the slider.
"""


def build_arg_parser():
    parser = base_demo.build_arg_parser()
    parser.add_argument("--ocr_protocol", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--ocr_ip", type=str, default="localhost")
    parser.add_argument("--ocr_port", type=int, default=8000)
    parser.add_argument("--ocr_model_name", type=str, default="rednote-hilab/dots.ocr")
    parser.add_argument("--ocr_temperature", type=float, default=0.1)
    parser.add_argument("--ocr_top_p", type=float, default=1.0)
    parser.add_argument("--ocr_max_tokens", type=int, default=4096)
    return parser


def get_ocr_client():
    global ocr_client

    if ocr_client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OCR demo requires the openai package for the dots.ocr vLLM client. "
                "Install it with: pip install openai"
            ) from exc

        base_url = f"{args.ocr_protocol}://{args.ocr_ip}:{args.ocr_port}/v1"
        ocr_client = OpenAI(
            api_key=os.environ.get("API_KEY", "0"),
            base_url=base_url,
        )

    return ocr_client


def normalize_prompt_text(text):
    return " ".join((text or "").strip().split())


def strip_existing_ocr_suffix(prompt):
    prompt = (prompt or "").strip()
    if OCR_SEPARATOR in prompt:
        prompt = prompt.split(OCR_SEPARATOR, 1)[0].strip()
    return prompt


def clean_ocr_text(ocr_text):
    return " ".join((ocr_text or "").strip().split())


def compose_ocr_prompt(base_prompt, ocr_text, append_ocr=True):
    base_prompt = strip_existing_ocr_suffix(base_prompt)
    base_prompt = normalize_prompt_text(base_prompt)
    ocr_text = clean_ocr_text(ocr_text)

    if not append_ocr or not ocr_text:
        return base_prompt
    if base_prompt:
        return f'{base_prompt}{OCR_SEPARATOR}"{ocr_text}"'
    return f'OCR Text: "{ocr_text}"'


def extract_ocr_text(input_image):
    if input_image is None:
        raise ValueError("Please upload an input image before running OCR.")

    client = get_ocr_client()
    rgb_image = input_image.convert("RGB")
    b64_uri = pil_image_to_base64(rgb_image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": b64_uri},
                },
                {
                    "type": "text",
                    "text": OCR_INSTRUCTION,
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=args.ocr_model_name,
        max_completion_tokens=args.ocr_max_tokens,
        temperature=args.ocr_temperature,
        top_p=args.ocr_top_p,
    )
    ocr_text = response.choices[0].message.content or ""
    return clean_ocr_text(ocr_text)


def generate_caption_for_ui(input_image, existing_ocr_text, append_ocr):
    caption = base_demo.process_llava(input_image)
    merged_prompt = compose_ocr_prompt(caption, existing_ocr_text, append_ocr=append_ocr)
    status = "Generated a LLaVA caption and refreshed the OCR-aware prompt."
    return caption, merged_prompt, status


def extract_ocr_for_ui(input_image, base_prompt, append_ocr):
    ocr_text = extract_ocr_text(input_image)
    merged_prompt = compose_ocr_prompt(base_prompt, ocr_text, append_ocr=append_ocr)
    if ocr_text:
        status = "Extracted OCR text from the image and appended it to the prompt draft."
    else:
        status = "No OCR text was extracted from the image."
    return ocr_text, merged_prompt, status


def compose_prompt_for_ui(base_prompt, ocr_text, append_ocr):
    merged_prompt = compose_ocr_prompt(base_prompt, ocr_text, append_ocr=append_ocr)
    if merged_prompt:
        status = "Updated the OCR-aware prompt."
    else:
        status = "Prompt is empty. Add a caption, OCR text, or both."
    return merged_prompt, status


def run_ocr_enhanced_sr(
    input_image,
    base_prompt,
    ocr_text,
    merged_prompt,
    append_ocr,
    manual_prompt_override,
    auto_caption,
    auto_extract_ocr,
    positive_prompt,
    negative_prompt,
    num_inference_steps,
    scale_factor,
    cfg_scale,
    seed,
):
    if input_image is None:
        raise ValueError("Please upload an input image before running DiT4SR.")

    status_lines = []
    base_prompt = strip_existing_ocr_suffix(base_prompt)
    ocr_text = clean_ocr_text(ocr_text)
    merged_prompt = (merged_prompt or "").strip()

    if auto_caption and not normalize_prompt_text(base_prompt):
        base_prompt = base_demo.process_llava(input_image)
        status_lines.append("Generated a caption with LLaVA.")

    if append_ocr and auto_extract_ocr and not ocr_text:
        ocr_text = extract_ocr_text(input_image)
        if ocr_text:
            status_lines.append("Extracted OCR text from the input image.")
        else:
            status_lines.append("OCR did not return any text, so SR used the visual prompt only.")

    composed_prompt = compose_ocr_prompt(base_prompt, ocr_text, append_ocr=append_ocr)
    final_prompt = composed_prompt
    if manual_prompt_override and merged_prompt:
        final_prompt = merged_prompt
    if not final_prompt:
        raise ValueError("Prompt is empty. Provide a prompt or enable auto caption generation.")

    if manual_prompt_override and merged_prompt:
        status_lines.append("Used the manually edited OCR-aware prompt.")
    elif append_ocr and ocr_text:
        status_lines.append("Composed the final prompt by appending OCR text.")
    else:
        status_lines.append("Used the visual prompt without OCR text.")

    comparison = base_demo.process_sr(
        input_image,
        final_prompt,
        positive_prompt,
        negative_prompt,
        num_inference_steps,
        scale_factor,
        cfg_scale,
        seed,
    )
    status_lines.append("Finished OCR-enhanced super-resolution.")
    return comparison, base_prompt, ocr_text, final_prompt, "\n".join(status_lines)


def build_demo():
    example_images = sorted(glob.glob("examples/*.png"))

    with gr.Blocks(title="DiT4SR OCR Prompt Demo") as demo:
        with gr.Row():
            gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                input_image = base_demo.create_input_image_component()
                base_prompt = gr.Textbox(
                    label="Base Prompt",
                    value="",
                    lines=4,
                    placeholder="Generate a LLaVA caption or write your own base prompt here.",
                )
                ocr_text = gr.Textbox(
                    label="Extracted OCR Text",
                    value="",
                    lines=4,
                    placeholder='OCR text will appear here, e.g. Store sign or poster text.',
                )
                merged_prompt = gr.Textbox(
                    label="OCR-Augmented Prompt",
                    value="",
                    lines=6,
                    placeholder='This prompt follows the repo format, e.g. ... OCR Text: "OPEN 24 HOURS"',
                )

                with gr.Accordion("Prompt Controls", open=True):
                    append_ocr = gr.Checkbox(label="Append OCR text to the prompt", value=True)
                    manual_prompt_override = gr.Checkbox(label="Use OCR-augmented prompt box as manual override", value=False)
                    auto_caption = gr.Checkbox(label="Auto-generate caption if base prompt is empty", value=True)
                    auto_extract_ocr = gr.Checkbox(label="Auto-extract OCR if OCR text is empty", value=True)

                with gr.Accordion("Generation Options", open=False):
                    positive_prompt = gr.Textbox(
                        label="Positive Prompt",
                        value='Cinematic, perfect without deformations, ultra HD, camera, detailed photo, realistic maximum, 32k, Color.',
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value='motion blur, noisy, dotted, pointed, deformed, lowres, chaotic CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, watermark, signature, jpeg artifacts.',
                    )
                    cfg_scale = gr.Slider(
                        label="Classifier Free Guidance Scale",
                        minimum=0.1,
                        maximum=10.0,
                        value=7.0,
                        step=0.1,
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=100,
                        value=20,
                        step=1,
                    )
                    seed = gr.Slider(
                        label="Seed (-1 for random)",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        value=-1,
                    )
                    scale_factor = gr.Number(label="SR Scale", value=4, precision=0)

                gr.Examples(examples=[[image_path] for image_path in example_images], inputs=input_image)
            with gr.Column():
                comparison_slider = base_demo.create_comparison_slider()
                status_box = gr.Textbox(label="Status", lines=6, interactive=False)
                with gr.Row():
                    run_llava_button = gr.Button(value="Run LLaVA")
                    run_ocr_button = gr.Button(value="Extract OCR")
                with gr.Row():
                    compose_prompt_button = gr.Button(value="Compose OCR Prompt")
                    run_sr_button = gr.Button(value="Run OCR-Enhanced DiT4SR", variant="primary")
                gr.Markdown(WORKFLOW)

        run_llava_button.click(
            fn=generate_caption_for_ui,
            inputs=[input_image, ocr_text, append_ocr],
            outputs=[base_prompt, merged_prompt, status_box],
        )
        run_ocr_button.click(
            fn=extract_ocr_for_ui,
            inputs=[input_image, base_prompt, append_ocr],
            outputs=[ocr_text, merged_prompt, status_box],
        )
        compose_prompt_button.click(
            fn=compose_prompt_for_ui,
            inputs=[base_prompt, ocr_text, append_ocr],
            outputs=[merged_prompt, status_box],
        )
        run_sr_button.click(
            fn=run_ocr_enhanced_sr,
            inputs=[
                input_image,
                base_prompt,
                ocr_text,
                merged_prompt,
                append_ocr,
                manual_prompt_override,
                auto_caption,
                auto_extract_ocr,
                positive_prompt,
                negative_prompt,
                num_inference_steps,
                scale_factor,
                cfg_scale,
                seed,
            ],
            outputs=[comparison_slider, base_prompt, ocr_text, merged_prompt, status_box],
        )
    return demo


def main():
    global args

    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    base_demo.args = args
    base_demo.LLaVA_device, base_demo.dit4sr_device = base_demo.resolve_devices()
    base_demo.llava_agent = base_demo.LLavaAgent(
        base_demo.LLAVA_MODEL_PATH,
        base_demo.LLaVA_device,
        load_8bit=True,
        load_4bit=False,
    )
    base_demo.pipeline = base_demo.load_dit4sr_pipeline(args, base_demo.dit4sr_device)

    demo = build_demo()
    demo.queue().launch()


if __name__ == "__main__":
    main()
