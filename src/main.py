"""
CLI entry point for Phase 2/3: run a question through the full
Planner -> Explorer -> Verifier -> Synthesizer pipeline and print the
answer plus a step-by-step trace.

Usage:
    python -m src.main "Who invented the sulphide rich composite battery?"
"""
import sys

from src.orchestrator import Orchestrator


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.main "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    orchestrator = Orchestrator()
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
