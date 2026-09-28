"""
CLI entry point: run a question through the full pipeline.

Usage:
    python -m src.main "Who invented the sulphide rich composite battery?"
    python -m src.main --rl "Who invented the sulphide rich composite battery?"
        (uses the trained RL router instead of the fixed static sequence —
        requires data/rl/q_table.json; train it first with
        python -m src.rl.train --simulate)
"""
import argparse
from src.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Ask a question against the GraphMind-X pipeline.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--rl", action="store_true", help="Use the trained RL router instead of the static sequence")
    parser.add_argument("--q-table-path", default=None)
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"Question: {question}\n")

    orchestrator = Orchestrator(use_rl_router=args.rl, q_table_path=args.q_table_path)
    try:
        result = orchestrator.answer(question)
    finally:
        orchestrator.close()

    print("=" * 60)
    print("ANSWER:")
    print(result.answer)
    print("=" * 60)
    print("\nTRACE:")
    for i, step in enumerate(result.trace, 1):
        print(f"  {i}. [{step.agent}] {step.description}")
        if step.detail:
            print(f"     {step.detail}")

    if result.used_fallback:
        print("\n(Note: one or more steps used a fallback due to an LLM or data issue — see trace above.)")


if __name__ == "__main__":
    main()