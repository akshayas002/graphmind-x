"""
Train the RL router via Q-learning.
Usage:
    python -m src.rl.train --simulate --episodes 300
    python -m src.rl.train --episodes 50 --questions "Who invented X?" "What uses Y?"
"""
import argparse

from src.agents.graph_explorer_agent import GraphExplorerAgent
from src.agents.models import SubQuestion
from src.agents.verifier_agent import VerifierAgent
from src.graph.neo4j_client import Neo4jClient
from src.llm.ollama_client import OllamaClient
from src.rl.actions import legal_actions
from src.rl.environment import RouterEnvironment
from src.rl.logger import TransitionLogger
from src.rl.q_learning import QLearningAgent
from src.rl.simulated_agents import SimulatedExplorer, SimulatedVerifier
from src import config

DEFAULT_SIMULATED_SCENARIOS = ["easy", "needs_one_retry", "unrecoverable"]
DEFAULT_REAL_QUESTIONS = [
    "Who invented the sulphide rich composite battery patent?",
    "Which company owns the most patents in this graph?",
    "What classification codes are associated with battery patents?",
]


def train_simulated(episodes, log_path, q_table_path, seed=None):
    agent = QLearningAgent(alpha=0.2, gamma=0.9, epsilon=0.3, seed=seed)
    logger = TransitionLogger(log_path)
    sq = SubQuestion(id="sq1", text="simulated question")

    for ep in range(episodes):
        scenario = DEFAULT_SIMULATED_SCENARIOS[ep % len(DEFAULT_SIMULATED_SCENARIOS)]
        explorer = SimulatedExplorer(scenario)
        env = RouterEnvironment(explorer, SimulatedVerifier(), sq, {})
        state = env.reset()
        done = False
        step = 0
        while not done:
            actions = legal_actions(state)
            action = agent.choose_action(state.as_tuple(), actions)
            next_state, reward, done, info = env.step(action)
            next_actions = legal_actions(next_state)
            agent.update(state.as_tuple(), action, reward, next_state.as_tuple(), next_actions, done)
            logger.log(ep, step, state, action, reward, next_state, done)
            state = next_state
            step += 1

        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {ep + 1}/{episodes} ({scenario}) complete.")

    agent.save(q_table_path)
    print(f"\nSaved Q-table to {q_table_path}")
    print(f"Logged transitions to {log_path}")
    return agent


def train_real(episodes, questions, log_path, q_table_path, seed=None):
    llm_client = OllamaClient()
    neo4j_client = Neo4jClient()
    explorer = GraphExplorerAgent(llm_client, neo4j_client)
    verifier = VerifierAgent(llm_client)

    try:
        schema = neo4j_client.get_schema_summary()
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")
        print("Is it running? (docker compose up -d)")
        return None

    agent = QLearningAgent(alpha=0.2, gamma=0.9, epsilon=0.3, seed=seed)
    logger = TransitionLogger(log_path)

    for ep in range(episodes):
        question_text = questions[ep % len(questions)]
        sq = SubQuestion(id=f"ep{ep}", text=question_text)
        env = RouterEnvironment(explorer, verifier, sq, schema)
        state = env.reset()
        done = False
        step = 0
        while not done:
            actions = legal_actions(state)
            action = agent.choose_action(state.as_tuple(), actions)
            next_state, reward, done, info = env.step(action)
            next_actions = legal_actions(next_state)
            agent.update(state.as_tuple(), action, reward, next_state.as_tuple(), next_actions, done)
            logger.log(ep, step, state, action, reward, next_state, done)
            state = next_state
            step += 1

        print(f"Episode {ep + 1}/{episodes} ('{question_text[:50]}...') complete.")

    agent.save(q_table_path)
    neo4j_client.close()
    print(f"\nSaved Q-table to {q_table_path}")
    print(f"Logged transitions to {log_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train the RL router via Q-learning.")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--questions", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-path", default=str(config.ROOT_DIR / "data" / "rl" / "transitions.jsonl"))
    parser.add_argument("--q-table-path", default=str(config.ROOT_DIR / "data" / "rl" / "q_table.json"))
    args = parser.parse_args()

    if args.simulate:
        print(f"Training on {args.episodes} SIMULATED episodes (no Ollama/Neo4j needed)...")
        agent = train_simulated(args.episodes, args.log_path, args.q_table_path, seed=args.seed)
    else:
        questions = args.questions or DEFAULT_REAL_QUESTIONS
        print(f"Training on {args.episodes} REAL episodes against your live Ollama + Neo4j...")
        agent = train_real(args.episodes, questions, args.log_path, args.q_table_path, seed=args.seed)

    if agent is not None:
        print("\nLearned policy summary (state -> best action):")
        for state_str, summary in agent.policy_summary().items():
            print(f"  {state_str}: {summary}")


if __name__ == "__main__":
    main()