
import ollama
from ollama import ResponseError

MODEL = "deepseek-r1:7b"


SYSTEM_PROMPT = """
You are an advanced reasoning and problem-solving agent.

CORE OBJECTIVE:
Produce accurate, logically sound, and verifiable answers. Prioritize correctness over speed, confidence, or agreement.

REASONING:
- Break complex problems into clear, manageable steps.
- Identify the goal, constraints, assumptions, and relevant information.
- Distinguish facts, assumptions, deductions, and speculation.
- Evaluate alternative approaches when multiple solutions exist.
- Check logical consistency and verify intermediate results.
- Consider edge cases, exceptions, and potential failure modes.
- For mathematical problems, verify calculations and units.
- For technical problems, consider practical constraints and trade-offs.

FACTUAL ACCURACY:
- Never fabricate facts, evidence, references, calculations, or results.
- Do not accept user assumptions as true without sufficient justification.
- Correct incorrect premises respectfully and explain the relevant issue.
- Explicitly acknowledge uncertainty, ambiguity, or insufficient information.
- Distinguish established facts from plausible hypotheses.
- Use available tools to verify information when necessary.
- Never claim to have executed, tested, searched, or verified something unless you actually did.

PROBLEM SOLVING:
- Prefer solutions that are correct, practical, and maintainable.
- Compare alternatives using relevant criteria rather than personal preference.
- Explain important limitations, risks, and trade-offs.
- If information is missing, ask a focused clarification or state reasonable assumptions.
- If a problem cannot be solved reliably, explain why and identify what is needed.

RESPONSE FORMAT:
- Begin with the answer or main conclusion when appropriate.
- Present a concise, understandable explanation supporting the conclusion.
- Use equations, examples, code, or structured lists when useful.
- Include important caveats without overwhelming the response.
- Adapt detail to the complexity of the problem.
- Avoid repetition, irrelevant discussion, and unsupported certainty.

INTEGRITY:
- Never prioritize pleasing the user over factual correctness.
- Do not invent missing information to make an answer appear complete.
- Revise conclusions when new evidence contradicts earlier assumptions.
- Present useful explanations and conclusions rather than unnecessary internal deliberation.

Your responsibility is to reason carefully, challenge unsupported assumptions,
verify what can be verified, and communicate conclusions honestly.
"""


class ReasoningAgent:
    def __init__(
        self,
        model: str = MODEL,
        host: str = "http://localhost:11434",
        max_history_messages: int = 30,
    ):
        self.model = model
        self.client = ollama.Client(host=host)
        self.max_history_messages = max_history_messages
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset(self):
        """Clear conversation history."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _trim_history(self):
        """Keep recent context while preserving complete user/assistant turns."""
        limit = max(2, self.max_history_messages)
        recent = self.messages[1:][-limit:]

        # Start with a user message, not an orphaned assistant response.
        while recent and recent[0]["role"] != "user":
            recent.pop(0)

        self.messages = [self.messages[0], *recent]

    def reason(self, problem: str, stream: bool = True) -> str:
        """Analyze a problem and return a reasoned answer."""
        problem = problem.strip()

        if not problem:
            return "Please provide a problem to analyze."

        self.messages.append({
            "role": "user",
            "content": problem,
        })
        self._trim_history()

        try:
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                stream=stream,
                options={
                    "temperature": 0.2,
                    "num_ctx": 8192,
                },
            )

            if stream:
                chunks = []

                for chunk in response:
                    text = chunk.message.content or ""
                    print(text, end="", flush=True)
                    chunks.append(text)

                print()
                answer = "".join(chunks)
            else:
                answer = response.message.content or ""
                print(answer)

            self.messages.append({
                "role": "assistant",
                "content": answer,
            })
            self._trim_history()

            return answer

        except ResponseError as exc:
            if self.messages[-1]["role"] == "user":
                self.messages.pop()

            error = f"Ollama error: {exc.error}"
            print(error)
            return error

        except Exception as exc:
            if self.messages[-1]["role"] == "user":
                self.messages.pop()

            error = f"Reasoning agent error: {exc}"
            print(error)
            return error


# def main():
#     agent = ReasoningAgent()

#     print(f"Reasoning Agent | Model: {agent.model}")
#     print("Type /reset to clear history or /exit to quit.\n")

#     while True:
#         try:
#             problem = input("Problem: ").strip()

#             if problem.lower() in {"/exit", "/quit"}:
#                 print("Goodbye!")
#                 break

#             if problem.lower() == "/reset":
#                 agent.reset()
#                 print("Conversation history cleared.\n")
#                 continue

#             if not problem:
#                 continue

#             print("\nAnalysis:\n")
#             agent.reason(problem, stream=True)
#             print()

#         except (KeyboardInterrupt, EOFError):
#             print("\nGoodbye!")
#             break


def main():
    agent = ReasoningAgent()

    queries = [
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain why.",
        "All engineers are problem-solvers. Alex is a problem-solver. Does that mean Alex is an engineer?",
        "If it rains, the ground gets wet. The ground is wet. Can we conclude that it rained?",
        "Every A is B, and every B is C. What can we conclude about A and C?",
        "Three boxes are labeled Apples, Oranges, and Mixed. Every label is incorrect. You may pick one fruit from one box. How can you correctly label all three boxes?",
        "A person says, Everything I say is false. Is this statement logically consistent?",
        "If P implies Q, and Q is false, what can we conclude about P?",
        "Five people are standing in a line. Arun is before Bala, Bala is before Charan, and Deepak is after Charan. What ordering constraints can we establish?",
        "A product costs ₹2000. Its price increases by 20% and then decreases by 20%. What is its final price?",
        "Solve: 3x + 7 = 31.",
        "A car travels 60 km at 30 km/h and another 60 km at 60 km/h. Calculate its average speed.",
        "A tank is filled by pipe A in 6 hours and pipe B in 3 hours. How long will they take together?",
        "If 2^x = 64, find x.",
        "A shopkeeper marks an item 40% above cost and gives a 10% discount. What is the profit percentage?",
        "Two fair dice are rolled. What is the probability that their sum is 7?",
        "A population grows by 10% annually. Starting at 1000, how many people will there be after two years?",
        "You have ₹10000. You spend 30% on equipment, 20% of the remaining money on software, and ₹1000 on transport. How much remains?",
        "Three workers finish a job in 12 days. Assuming equal productivity, how long would six workers take?",
        "A robot must visit five locations while minimizing travel distance. What information and algorithm would you need to determine the optimal route?",
        "You have 8 GB of RAM available, but a model requires 12 GB. Propose three practical ways to run it and explain the trade-offs.",
        "A server becomes slower as concurrent users increase. Give a systematic debugging procedure to identify the bottleneck.",
        "A project has tasks A, B, C, D, and E. B depends on A, C depends on A, D depends on B and C, and E depends on D. Identify the dependency paths and explain what additional information is needed to determine the critical path.",
        "You have 30 minutes and five tasks with different priorities, durations, and deadlines. How would you prioritize them?",
        "An algorithm works for small inputs but times out for large inputs. Explain how you would diagnose and improve it.",
        "Find the next number: 2, 6, 12, 20, 30, ?",
        "Find the next number: 1, 1, 2, 3, 5, 8, ?",
        "Find the odd one out: 16, 25, 36, 49, 63, 64.",
        "Complete the sequence: A, C, F, J, O, ?",
        "If BOOK becomes CPPL by shifting every letter forward by one, how does GAME transform?",
        "A sequence starts at 1, and each next term is obtained by doubling the previous term and adding 1. Write the first six terms.",
        "A dataset has a high mean but a low median. What might this suggest about its distribution?",
        "A model's training performance improves while its performance on unseen data worsens. What likely problem does this indicate?",
        "Three indivisible jobs require 2, 3, and 4 hours. Two workers can work simultaneously. Find a schedule minimizing the total completion time.",
        "Four people must cross a bridge at night with one flashlight. Their crossing times are 1, 2, 7, and 10 minutes. At most two people can cross at once. Find the minimum total time.",
        "A Sudoku row contains the digits 1 through 9, with eight distinct digits already present. What number belongs in the empty cell?",
        "You must schedule three exams so that students taking overlapping subjects never have simultaneous exams. What information is needed, and how could you model the problem?",
        "A delivery agent must visit six destinations and return to its starting point. Explain why finding the shortest route can be difficult.",
        "You have five projects but resources for only two. Explain a rational selection strategy using value, cost, risk, and dependencies.",
        "A robot must navigate a grid with obstacles. Compare BFS, Dijkstra's algorithm, and A* for finding a path.",
        "You must choose between two models: one is more accurate but uses four times as much memory. How would you decide which is better for deployment?",
        "Ice cream sales and drowning incidents both increase in summer. Does this prove that ice cream causes drowning? Explain.",
        "A website's conversion rate increased after a redesign. How would you determine whether the redesign caused the improvement?",
        "A machine learning model achieves 99% accuracy on an imbalanced dataset. Why might this result be misleading?",
        "Someone claims that a treatment works because ten people improved after taking it. Evaluate the strength of this evidence.",
        "Two events happen together repeatedly. What additional evidence would help establish a causal relationship?",
        "A benchmark shows that Model A is faster than Model B on one task. Can we conclude that A is better overall?",
        "A program passed all five test cases provided by its author. Is it necessarily correct? Explain.",
        "An AI agent gives a detailed explanation for its answer. Does the explanation guarantee that the answer is correct?",
        "Evaluate this claim: All prime numbers are odd, and 2 is prime. Identify the error and correct the claim.",
        "A program calculates the average of 10, 20, and 30 as 25. Identify the error and calculate the correct average."
    ]

    print(f"Reasoning Agent | Model: {agent.model}")
    print(f"Running {len(queries)} predefined reasoning tests.\n")

    try:
        for index, query in enumerate(queries, start=1):
            print("=" * 80)
            print(f"TEST {index}/{len(queries)}")
            print(f"Problem: {query}\n")
            print("Analysis:\n")

            agent.reason(query, stream=True)
            print("\n")

    except (KeyboardInterrupt, EOFError):
        print("\nTest run interrupted.")

    finally:
        print(f"\nCompleted {index if 'index' in locals() else 0}/{len(queries)} tests.")
        print("Reasoning benchmark finished.")


if __name__ == "__main__":
    main()
