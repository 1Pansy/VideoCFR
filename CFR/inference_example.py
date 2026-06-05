import argparse
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', "
    "'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
    "It is encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your "
    "final answer between the <answer> and </answer> tags."
)


def parse_args():
    default_video = Path(__file__).resolve().parent / "example_video" / "video1.mp4"
    parser = argparse.ArgumentParser(description="Run VideoCFR-compatible single-video inference with vLLM.")
    parser.add_argument("--model_path", default="Video-R1/Video-R1-7B", help="Local path or Hugging Face model id.")
    parser.add_argument("--video_path", default=str(default_video), help="Path to the input video.")
    parser.add_argument(
        "--question",
        default="Which moving object in the video loses system energy?",
        help="Question to ask about the video.",
    )
    parser.add_argument(
        "--problem_type",
        default="free-form",
        choices=sorted(TYPE_TEMPLATE),
        help="Answer type used to build the prompt.",
    )
    parser.add_argument("--nframes", type=int, default=32, help="Maximum number of video frames.")
    parser.add_argument("--max_pixels", type=int, default=200704, help="Maximum pixels for each frame.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--max_model_len", type=int, default=81920, help="Maximum model context length.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU memory fraction.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = str(Path(args.video_path).expanduser())

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"video": 1, "image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": args.max_pixels,
                    "nframes": args.nframes,
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=args.question) + TYPE_TEMPLATE[args.problem_type],
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    llm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs[0]},
            "mm_processor_kwargs": {key: val[0] for key, val in video_kwargs.items()},
        }
    ]

    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
