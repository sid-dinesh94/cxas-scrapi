# Issue: Implement PRD-Conditioned Dojo Simulator & Sensei Evaluator Framework

## Background & Motivation
In the GECX/CXAS conversational agent lifecycle, relying on static YAML test cases and rigid `user_goals` creates several development bottlenecks:
1. **Lack of Diversity:** Static test cases do not capture the realistic variance of human dialogue, edge cases, or difficult user personalities (e.g. rambling, impatient, or adversarial callers).
2. **Maintenance Overhead:** As the PRD evolves, manually updating hundreds of turn-by-turn expectation scripts becomes unsustainable.
3. **Shallow Vetting:** Turn-by-turn evaluation does not adequately assess overall goal satisfaction, compliance with subtle brand guidelines, or complex multi-tool call graphs.

## Proposed Architecture (Dojo & Sensei)
To address these bottlenecks, we introduce the **Dojo & Sensei Framework**—a domain-specific implementation of the Gemini Gym evaluation pattern:

```
[PRD / Agent Instructions] 
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Persona Foundry                                        │
│ Extracts diverse Scenario Matrix (Happy, Edge, Diff)   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ The Dojo (Simulated Users)                             │
│ Runs autonomous multi-turn simulation loop             │
│ (Powered by gemini-3.1-flash-lite-preview)             │
└────────────────────────────────────────────────────────┘
       │
       ▼ (Completed Transcript)
┌────────────────────────────────────────────────────────┐
│ Sensei Evaluator                                       │
│ Deferred post-session grading against PRD rubric       │
│ (Powered by gemini-3.1-pro-preview)                    │
└────────────────────────────────────────────────────────┘
```

### 1. The Dojo (PRD-Conditioned User Simulation)
* **Persona Foundry:** Parses the raw agent instructions (`agent_instruction.md`) and generates a structured `ScenarioMatrix` exploring Happy Paths, Edge Cases, and Difficult Users.
* **Autonomous Simulation:** Gemini acts as the customer/guest and converses with the target agent via `sessions_client.run()` until `<DOJO_GOAL_ACHIEVED>` or `<DOJO_GOAL_FAILED>` is reached.

### 2. The Sensei (Hill-Climbing Evaluator)
* **Deferred Evaluation:** Decoupled from the conversation loop to prevent exponential token growth.
* **Structured Output Rubric:** Emits a Pydantic-enforced JSON rubric containing boolean success, float scores (0-100), detailed findings, and actionable prompt engineering recommendations.
* **Rule Coverage Extraction:** Audits the transcript against extracted PRD operational rules to report exact rule coverage.

## Token Optimizations & Mimicking Fixes Implemented
1. **Customer Persona Enforcement:** Explicitly instructs the Persona Foundry to generate a profile for a **CUSTOMER / GUEST**, eliminating the bug where the simulator adopted the host agent's persona.
2. **Aggressive Anti-Mimicry Constraints:** Enforces strict rules against assistant-speak, bullet points, and paragraph verbosity (capping responses at 1–2 sentences).
3. **Sliding Window Transcript:** Caps quadratic context growth on long simulations by retaining the initial goal but maintaining a 6-turn sliding window during the dialogue loop.
4. **Model Tiering:** Uses `gemini-3.1-flash-lite-preview` for the simulated users while reserving `gemini-3.1-pro-preview` for Sensei.

## Relevant Files Included
* `.agents/skills/cxas-agent-foundry/scripts/scrapi-dojo.py` (Core simulation runner)
* `examples/bella_notte/scenario_matrix.json` (Cached extracted scenarios)
* `examples/bella_notte/dojo_report.html` / `dojo_gym_report.html` (Rich HTML evaluation reports)
