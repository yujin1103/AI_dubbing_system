#!/usr/bin/env python3
"""instruct2 TRT 포팅 검증 — prebuilt TRT-LLM 엔진에서 instruct2 입력이
유효한 speech token 시퀀스를 생성하는지 확인.

포팅 핵심(frontend_instruct2 의미 = TRT-LLM 입력):
  zero-shot  : <|sos|> + prompt_text(참조전사) + target_text + <|task_id|> + prompt_speech_tokens
  instruct2  : <|sos|> + instruct_text(감정지시) + target_text + <|task_id|> + (빈값; prompt_speech_token 생략)
즉 reference_text→instruct_text, assistant(prompt_speech_tokens)→"" 만 바꾸면 됨.
"""
import argparse
import re

import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp

BASE = "/workspace/CosyVoice/runtime/triton_trtllm"
ENGINE = f"{BASE}/trt_engines_bfloat16"
TOK = f"{BASE}/hf_cosyvoice3_llm"


def parse_speech_ids(text: str):
    return [int(m) for m in re.findall(r"<\|s_(\d+)\|>", text)]


def build_input_ids(tokenizer, prompt_or_instruct_text: str, target_text: str,
                    prompt_speech_id_str: str = ""):
    """offline_inference.py 와 동일한 chat-template 입력 구성.

    instruct2: prompt_or_instruct_text=instruct_text, prompt_speech_id_str=""(생략).
    """
    full_text = prompt_or_instruct_text + target_text
    chat = [
        {"role": "user", "content": full_text},
        {"role": "assistant", "content": prompt_speech_id_str},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", continue_final_message=True
    )
    return input_ids.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="안녕하세요. 만나서 정말 반갑습니다.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    print(f"tensorrt_llm {tensorrt_llm.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(TOK)
    end_id = (tokenizer.convert_tokens_to_ids("<|eos1|>")
              if "<|eos1|>" in tokenizer.get_vocab() else tokenizer.eos_token_id)
    print(f"end_id(<|eos1|>)={end_id}")

    runner = ModelRunnerCpp.from_dir(
        engine_dir=ENGINE, rank=0, max_output_len=2048, max_batch_size=1,
        max_input_len=512, kv_cache_free_gpu_memory_fraction=0.5,
        cuda_graph_mode=False, gather_generation_logits=False,
    )

    cases = [
        ("instruct2=기쁨/흥분", "아주 기쁘고 들뜬 목소리로 말해주세요. "),
        ("instruct2=슬픔/낮은톤", "아주 슬프고 가라앉은 목소리로 말해주세요. "),
        ("대조: instruct 없음", ""),
    ]
    results = []
    for name, instruct in cases:
        ids = build_input_ids(tokenizer, instruct, args.target, "")
        out = runner.generate(
            batch_input_ids=[ids], max_new_tokens=2048, end_id=end_id, pad_id=end_id,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=1.1, num_return_sequences=1, streaming=False,
            output_sequence_lengths=True, return_dict=True,
            return_all_generated_tokens=False,
        )
        torch.cuda.synchronize()
        oid, slen = out["output_ids"], out["sequence_lengths"]
        gen = oid[0][0][ids.size(0):slen[0][0]].tolist()
        gen_text = tokenizer.decode(gen)
        sids = parse_speech_ids(gen_text)
        ok = len(sids) > 0
        results.append((name, ids.size(0), len(gen), len(sids), sids[:6], ok))
        print(f"[{name}] input_len={ids.size(0)} gen={len(gen)} "
              f"speech_ids={len(sids)} first6={sids[:6]} OK={ok}")

    print("\n=== 검증 요약 ===")
    all_ok = all(r[5] for r in results)
    seqs = [tuple(r[4]) for r in results]
    distinct = len(set(seqs)) > 1
    print(f"모든 케이스 유효 speech token 생성: {all_ok}")
    print(f"instruct별 출력 상이(instruct가 LLM에 영향): {distinct}")
    print(f"=> instruct2 TRT 포팅 {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
