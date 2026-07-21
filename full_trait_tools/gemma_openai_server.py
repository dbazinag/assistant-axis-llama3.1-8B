#!/usr/bin/env python3
"""
gemma_openai_server.py

Minimal OpenAI-compatible chat-completions server for Gemma-4-31B-it, so HarmBench
query-based attacks (GPTFuzz/PAIR/TAP) can use Gemma as a *target* without loading
it inside HarmBench's vLLM/fastchat stack (neither supports gemma4). The model runs
here in the proven .venv_gemma recipe (transformers 5.8, trust_remote_code, sdpa
under chunked_sdpa_scope); HarmBench just points an OpenAI client at this endpoint.

Stdlib only (http.server) — no new deps in .venv_gemma. GPU access is serialised
with a lock (one model, one GPU); concurrent requests queue.

Endpoints:
  GET  /health                 -> {"status": "ok", "model_loaded": bool}
  POST /v1/chat/completions    -> OpenAI-compatible:
      request : {"model", "messages":[{"role","content"}], "temperature", "max_tokens", "n"}
      response: {"choices":[{"index","message":{"role":"assistant","content"},"finish_reason"}], ...}

Usage:
  .venv_gemma/bin/python full_trait_tools/gemma_openai_server.py \
      --model /path/to/gemma-4-31B-it --port 8000
"""

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma_server")

# Globals set in main()
MODEL = None
TOKENIZER = None
DEVICE = None
GEN_LOCK = threading.Lock()       # serialise GPU generation
MAX_NEW_TOKENS_CAP = 1024


def build_prompt_ids(messages):
    """Turn OpenAI-style messages into Gemma input_ids.

    Gemma-4's chat template has user/assistant turns (no dedicated system role).
    Any system content is folded into the first user turn so the target sees a
    well-formed single-user prompt (matching the HB/WJB Gemma collectors).
    """
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system" and m.get("content")]
    convo = []
    folded_system = "\n\n".join(system_parts).strip()
    pending_system = folded_system
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "system":
            continue
        if role == "user" and pending_system:
            content = f"{pending_system}\n\n{content}".strip()
            pending_system = ""
        convo.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    if not convo:                                  # only a system message -> treat as user
        convo = [{"role": "user", "content": folded_system}]
    text = TOKENIZER.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    return TOKENIZER(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)


def generate_one(messages, temperature, max_tokens):
    input_ids = build_prompt_ids(messages)
    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=min(int(max_tokens or 512), MAX_NEW_TOKENS_CAP),
        do_sample=do_sample,
        pad_token_id=TOKENIZER.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
    with GEN_LOCK:
        with torch.no_grad(), chunked_sdpa_scope():
            out = MODEL.generate(input_ids=input_ids, **gen_kwargs)
    resp_ids = out[0, input_ids.shape[1]:]
    return TOKENIZER.decode(resp_ids, skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # quiet default logging

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok", "model_loaded": MODEL is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            messages = req.get("messages", [])
            temperature = req.get("temperature", 0.0)
            max_tokens = req.get("max_tokens", 512)
            n = int(req.get("n", 1) or 1)
            t0 = time.time()
            choices = []
            for i in range(n):
                content = generate_one(messages, temperature, max_tokens)
                choices.append({
                    "index": i,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                })
            logger.info(f"chat.completions n={n} temp={temperature} -> {time.time()-t0:.1f}s")
            self._send(200, {
                "id": "chatcmpl-gemma",
                "object": "chat.completion",
                "model": req.get("model", "gemma-4-31b-it"),
                "choices": choices,
            })
        except Exception as e:
            logger.warning(f"request failed: {e}")
            self._send(500, {"error": str(e)})


def main():
    global MODEL, TOKENIZER, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU available.")
    DEVICE = torch.device("cuda:0")

    logger.info(f"Loading {args.model} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "left"
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    for p in MODEL.parameters():
        p.requires_grad_(False)
    logger.info("Model loaded. Serving.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(f"Listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
