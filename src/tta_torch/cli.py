"""CLI interface for TTA-Torch self-tuning inference."""

import argparse
import gc
import logging
import sys
import time

import torch

from .engine import TTAModel
from .loader import load_tta_model

logger = logging.getLogger("tta_torch")

METHODS = ["raw", "confidence_gated", "best_of_n", "majority", "entropy_weighted"]

SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "What is 137 * 29?",
    "Name the largest planet in our solar system.",
    "What year was the Magna Carta signed?",
    "How many sides does a hexagon have?",
    "What is the chemical symbol for gold?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the square root of 144?",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tta-torch",
        description="TTA-Torch: Self-Tuning Inference Engine",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run TTA generation on a prompt")
    gen.add_argument("--prompt", "-p", type=str, required=True, help="Input prompt")
    gen.add_argument("--model", "-m", type=str, default="gpt2", help="Model ID or path")
    gen.add_argument(
        "--method",
        type=str,
        default="raw",
        choices=METHODS,
        help="TTA method to use",
    )
    gen.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    gen.add_argument(
        "--entropy-threshold", type=float, default=1.5, help="Entropy threshold"
    )
    gen.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    gen.add_argument("--inner-steps", type=int, default=5, help="Inner optimization steps")
    gen.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    gen.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen.add_argument(
        "--n-samples", type=int, default=5, help="Number of samples for best_of_n"
    )

    bench = sub.add_parser("benchmark", help="Run a quick accuracy benchmark")
    bench.add_argument("--model", "-m", type=str, default="gpt2", help="Model ID or path")
    bench.add_argument(
        "--method",
        type=str,
        default="raw",
        choices=METHODS,
        help="TTA method to use",
    )
    bench.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    bench.add_argument(
        "--entropy-threshold", type=float, default=1.5, help="Entropy threshold"
    )
    bench.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    bench.add_argument("--inner-steps", type=int, default=5, help="Inner optimization steps")
    bench.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens")
    bench.add_argument(
        "--n-samples", type=int, default=5, help="Number of samples for best_of_n"
    )

    sub.add_parser("clean", help="Clear GPU memory cache")

    return parser


def cmd_generate(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_tta_model(args.model, lora_rank=args.lora_rank)
    tta_config = {
        "entropy_threshold": args.entropy_threshold,
        "learning_rate": args.learning_rate,
        "inner_steps": args.inner_steps,
        "max_new_tokens": args.max_new_tokens,
        "verbose": args.verbose,
    }
    tta = TTAModel(model, tta_config)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    logger.info("Generating with method=%s, max_new_tokens=%d", args.method, args.max_new_tokens)

    start = time.perf_counter()
    if args.method == "confidence_gated":
        output, reason, entropy = tta.generate_confidence_gated(
            input_ids, n_passes=args.n_samples, temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Reason: {reason}, Entropy: {entropy:.4f}")
    elif args.method == "majority":
        answer, all_answers = tta.generate_majority(
            input_ids, tokenizer=tokenizer, n_passes=args.n_samples,
            temperature=args.temperature, max_tokens=args.max_new_tokens,
        )
        result = str(answer) + " (answers: " + str(all_answers) + ")"
    elif args.method == "best_of_n":
        output = tta.generate_best_of_n(input_ids, max_tokens=args.max_new_tokens)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    elif args.method == "entropy_weighted":
        output, reason, entropy = tta.generate_entropy_weighted_vote(
            input_ids, n_passes=args.n_samples, temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Reason: {reason}, Entropy: {entropy:.4f}")
    else:
        output = tta.generate(input_ids, max_tokens=args.max_new_tokens)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    elapsed = time.perf_counter() - start

    print("\n--- Generated Output ---")
    print(result)
    print(f"\n--- Stats ---")
    print(f"Method:        {args.method}")
    print(f"Wall time:     {elapsed:.3f}s")


def cmd_benchmark(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_tta_model(args.model, lora_rank=args.lora_rank)
    tta_config = {
        "entropy_threshold": args.entropy_threshold,
        "learning_rate": args.learning_rate,
        "inner_steps": args.inner_steps,
        "max_new_tokens": args.max_new_tokens,
        "verbose": args.verbose,
    }
    tta = TTAModel(model, tta_config)

    total = len(SAMPLE_QUESTIONS)
    logger.info("Benchmarking %d questions with method=%s", total, args.method)

    correct = 0
    total_time = 0.0

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        start = time.perf_counter()
        if args.method == "confidence_gated":
            output, reason, entropy = tta.generate_confidence_gated(
                input_ids, n_passes=args.n_samples, temperature=0.3,
                max_tokens=args.max_new_tokens,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        elif args.method == "majority":
            answer, _ = tta.generate_majority(
                input_ids, tokenizer=tokenizer, n_passes=args.n_samples,
                temperature=0.3, max_tokens=args.max_new_tokens,
            )
            text = str(answer)
        elif args.method == "best_of_n":
            output = tta.generate_best_of_n(input_ids, max_tokens=args.max_new_tokens)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        elif args.method == "entropy_weighted":
            output, _, _ = tta.generate_entropy_weighted_vote(
                input_ids, n_passes=args.n_samples, temperature=0.3,
                max_tokens=args.max_new_tokens,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            output = tta.generate(input_ids, max_tokens=args.max_new_tokens, temperature=0.3)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        answered = len(text.strip()) > 0
        if answered:
            correct += 1
        print(f"  Q{i}: {question[:50]}")
        print(f"    A: {text[:100]}")

    accuracy = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0

    print(f"\n--- Benchmark Results ---")
    print(f"Method:        {args.method}")
    print(f"Answered:      {correct}/{total}")
    print(f"Accuracy:      {accuracy:.1%}")
    print(f"Total time:    {total_time:.3f}s")
    print(f"Avg per query: {avg_time:.3f}s")


def cmd_clean(_args: argparse.Namespace) -> None:
    logger.info("Cleaning GPU memory cache")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"GPU memory cleaned. Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
    else:
        print("No CUDA device found. GC collected unreferenced objects only.")


COMMANDS = {
    "generate": cmd_generate,
    "benchmark": cmd_benchmark,
    "clean": cmd_clean,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        COMMANDS[args.command](args)
    except Exception as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
