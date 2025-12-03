from vllm import LLM, SamplingParams
from typing import List
from tqdm import tqdm
import json
import argparse
import torch
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
import json
import time

class NegativeCaptionLLMParser:
    def __init__(self,
                 model_id: str,
                 caption_file_path: str,
                 tp_size: int = 1,
                 concurrency: int = 1,
                 batch_size: int = 4
            ):

        print(f"Loading vLLM model: {model_id}")

        self.config = vLLMEngineProcessorConfig(
            model_source=model_id,  
            engine_kwargs={
                "tensor_parallel_size": tp_size,   
                "max_model_len": 2048,
                "enable_chunked_prefill": True,
                "gpu_memory_utilization": 0.95,
            },
            concurrency=concurrency,      
            batch_size=batch_size        
        )

        with open(caption_file_path, "r") as f:
            data = json.load(f)

        self.captions = data["annotations"]

        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
        )
    @staticmethod
    def _base_prompt():
        base_prompt = """
            You are given an image caption. Generate 1–5 HARD NEGATIVE captions.

            A hard negative:
            - Is similar in length and structure to the original.
            - Changes a few key details (colors, numbers, objects, spatial relations, or who does what).
            - Must be clearly wrong for the original image, even if it looks close.
            - Do NOT just paraphrase or add meaningless adjectives.

            Focus on EXCHANGING attributes within the caption (e.g., swapping colors, left/right, roles).

            Example:
            Input: A dog to the left of the cat.
            Output: A dog to the right of the cat.

            Input: A person wearing a red helmet drives a motorbike on a dirt road.
            Output: A person wearing a blue helmet drives a motorbike on a snowy road.

            IMPORTANT:
            Return ONLY valid JSON in this exact format:
            {"negative_captions": ["caption1", "caption2", "caption3", "caption4", "caption5"]}
            Use between 1 and 5 captions in the list. No explanations.
        """
        return base_prompt
    
    
    def _preprocess(row):
        caption = row["caption"]
        return dict(
            messages=[
                {'role': 'user', 'content': NegativeCaptionLLMParser._base_prompt() + f"caption is {caption}"}
            ],
            sampling_params=dict(
                temperature=0.0,
                max_tokens=512,
            )
        )

    def _postprocess(row):
        raw_text = row["generated_text"]

        try:
            json_part = raw_text[raw_text.index("{"): raw_text.rindex("}") + 1]
            neg = json.loads(json_part)
        except Exception as e:
            neg = {"negative_captions": []}

        return {
            "id": row["id"],
            "image_id": row["image_id"],
            "caption": row["caption"],
            "negative_captions": neg["negative_captions"],
        }
    
    def parse(self, limit: int, output_path: str):
        # all_outputs = {}

        processor = build_llm_processor(
            self.config,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
        )
        
        start_time = time.time()
        print('start inference...')
        results = processor(self.captions[:limit])
        
        print(f"total time is:, {time.time() - start_time:.2f}s")

        with open(output_path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create negative captions using vLLM")
    parser.add_argument("--model_id", type=str,
                        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT8")
    parser.add_argument("--caption_file_path", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="number of gpu")
    parser.add_argument("--concurrency", type=int, default=1, help="number of replicas")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size per replica")

    args = parser.parse_args()

    llm_parser = NegativeCaptionLLMParser(
        model_id=args.model_id,
        caption_file_path=args.caption_file_path,
        dtype=args.torch_dtype,
        tp_size=args.tensor_parallel_size,
        concurrency=args.concurrency,
        batch_size=args.batch_size
    )

    llm_parser.parse(limit=args.limit, output_path=args.output_path)
