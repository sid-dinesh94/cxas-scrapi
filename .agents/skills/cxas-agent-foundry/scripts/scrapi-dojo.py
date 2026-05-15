#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dojo & Sensei dynamic simulation runner using cxas-scrapi.

This script handles PRD-driven dynamic simulations and automated grading.
"""

import argparse
import json
import os
import sys
import time
import uuid
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional

import pydantic
from google import genai

from cxas_scrapi.evals.simulation_evals import (
    SimulationEvals,
    StepStatus,
)
from config import load_app_name, resolve_project_dir

_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


class PerplexityProxyOutput(pydantic.BaseModel):
    perplexity_score: float
    justification: str


class DojoSimRunner(SimulationEvals):
    """Runner for dynamic Dojo simulations."""

    def _compute_perplexity_proxy(self, context: str, agent_response: str, model: str) -> float:
        """Compute an experimental perplexity proxy score using Gemini-as-a-Judge."""
        prompt = f"""You are an advanced AI language model evaluation judge specializing in information theory and perplexity analysis.
Your task is to evaluate the "perplexity proxy score" of an AI Agent's response given the preceding conversation context and raw states.

Preceding Context:
{context}

Agent Response to Evaluate:
{agent_response}

Analyze how predictable, fluent, and statistically aligned the Agent's Response is relative to the context. 
Focus strictly on semantic relevance and natural conversational flow. Crucially, ignore standard system headers, legal disclaimers, welcome boilerplate, or automated tool instructions when calculating the perplexity proxy score.
Calculate a perplexity proxy score on a scale from 1.0 to 100.0+:
- 1.0 - 10.0: Highly predictable, perfectly fluent, completely standard response.
- 10.1 - 25.0: Moderately predictable, normal conversational variance.
- 25.1 - 50.0: High surprise factor, unexpected shift, or minor phrasing awkwardness.
- 50.1+: Extreme perplexity, nonsensical response, severe hallucination, or complete breakdown of context adherence."""

        try:
            output: PerplexityProxyOutput = self.genai_client.generate(
                prompt=prompt,
                model_name=model,
                response_mime_type="application/json",
                response_schema=PerplexityProxyOutput,
            )
            if output and hasattr(output, "perplexity_score"):
                return float(output.perplexity_score)
        except Exception:
            pass
        return 10.0 # Default fallback safe baseline

    def _parse_prd_to_matrix(self, doc_path: str, model: str) -> Dict[str, Any]:
        """Parse PRD/instructions and generate a structured Scenario Matrix."""
        import hashlib
        
        with open(doc_path, "r") as f:
            instructions = f.read()
            
        file_hash = hashlib.sha256(instructions.encode('utf-8')).hexdigest()
        
        prompt = f"""You are an advanced AI engineer designing a suite of simulation tests for a conversational agent.
Read the following agent instructions and extract a diverse matrix of test scenarios to evaluate the agent.

Agent Instructions:
{instructions}

You must generate a list of at least 10 diverse scenarios exploring a wide variety of call paths:
- Multiple "Happy Path" scenarios with different data.
- Multiple "Edge Case" scenarios testing specific rules (e.g., large party size requiring credit card).
- Multiple "Difficult User" scenarios (e.g., long conversers who only give little bits of information at a time, users who ask many off-topic questions, or users who are impatient).

Return the result as a JSON object matching this schema:
{{
  "scenarios": [
    {{
      "id": "SCENARIO_01",
      "type": "Happy Path" / "Edge Case" / "Difficult User",
      "title": "Short descriptive title",
      "description": "Detailed description of what the scenario tests",
      "goal": "The high-level goal the user wants to achieve",
      "hidden_agendas": ["List of specific behaviors or constraints the user must follow"]
    }}
  ]
}}
"""
        try:
            class Scenario(pydantic.BaseModel):
                id: str
                type: str
                title: str
                description: str
                goal: str
                hidden_agendas: List[str] = []
                
            class ScenarioMatrix(pydantic.BaseModel):
                scenarios: List[Scenario]
                
            output: ScenarioMatrix = self.genai_client.generate(
                prompt=prompt,
                model_name=model,
                response_mime_type="application/json",
                response_schema=ScenarioMatrix,
            )
            
            if output:
                matrix_data = {
                    "hash": file_hash,
                    "scenarios": [s.model_dump() for s in output.scenarios]
                }
                
                doc_dir = os.path.dirname(doc_path)
                matrix_path = os.path.join(doc_dir, "scenario_matrix.json")
                with open(matrix_path, "w") as f:
                    json.dump(matrix_data, f, indent=2)
                    
                print(f"Saved scenario matrix with {len(output.scenarios)} scenarios to {matrix_path}")
                return matrix_data
                
        except Exception as e:
            print(f"Error parsing PRD to matrix: {e}")
        return None

    def _load_scenario_matrix(self, doc_path: str, model: str) -> Dict[str, Any]:
        """Load scenario matrix from file or regenerate if hash mismatch."""
        import hashlib
        
        doc_dir = os.path.dirname(doc_path)
        matrix_path = os.path.join(doc_dir, "scenario_matrix.json")
        
        with open(doc_path, "r") as f:
            instructions = f.read()
        current_hash = hashlib.sha256(instructions.encode('utf-8')).hexdigest()
        
        if os.path.exists(matrix_path):
            try:
                with open(matrix_path, "r") as f:
                    data = json.load(f)
                if data.get("hash") == current_hash:
                    print(f"Using cached scenario matrix from {matrix_path}")
                    return data
                else:
                    print("Hash mismatch. Regenerating scenario matrix...")
            except Exception as e:
                print(f"Error reading cached matrix: {e}. Regenerating...")
                
        return self._parse_prd_to_matrix(doc_path, model)

    def simulate_dynamic_conversation(
        self,
        persona_prompt: str,
        initial_utterance: str = "Hi",
        model: str = _DEFAULT_MODEL,
        session_id: Optional[str] = None,
        console_logging: bool = True,
    ) -> List[str]:
        """Run a dynamic conversation between the Dojo Agent and the target Agent."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        detailed_trace = []
        
        dojo_system_instruction = f"""You are a simulated CUSTOMER calling a business.
Here is your persona and goal:
{persona_prompt}

CRITICAL RULES:
1. You are the CUSTOMER. You are NOT the agent or assistant. Do NOT offer assistance, ask how you can help, or take reservations.
2. Keep your responses short and conversational (1-2 sentences max). Do NOT output bullet points or long paragraphs.
3. Do NOT mimic the agent's phrasing. If the agent asks a question, answer it from the perspective of a customer.
4. When your goal is achieved or if you give up, append <DOJO_GOAL_ACHIEVED> or <DOJO_GOAL_FAILED>.
"""
        
        user_utterance = initial_utterance
        detailed_trace.append(f"User: {user_utterance}")
        
        if console_logging:
            print(f"Dojo User: {user_utterance}")
            
        dojo_transcript = [f"User: {user_utterance}"]
        
        turn = 0
        max_turns = 20
        should_terminate = False
        
        while turn < max_turns:
            kwargs = {
                "session_id": session_id,
                "text": user_utterance,
                "modality": "text",
            }
            try:
                response = self.sessions_client.run(**kwargs)
            except Exception as e:
                print(f"Error calling target agent: {e}")
                break
                
            if not response:
                break
                
            agent_text, trace_chunks, session_ended = self._parse_agent_response(response)
            detailed_trace.append("\n".join(trace_chunks))
            
            if console_logging:
                print(f"Agent: {agent_text}")
                
            if session_ended:
                print("Session ended by target agent.")
                break
                
            dojo_transcript.append(f"Agent: {agent_text}")
            
            if should_terminate:
                if console_logging:
                    print("Dojo user signaled completion. Ending conversation.")
                break
                
            # Keep the initial user utterance, but only the last 6 turns (3 user/agent pairs) of dialogue
            window = [dojo_transcript[0]] + dojo_transcript[-6:] if len(dojo_transcript) > 7 else dojo_transcript
            dojo_prompt = f"""The conversation so far:
{"\n".join(window)}

Based on your persona and goal, what do you say next?
"""
            
            try:
                dojo_response = self.genai_client.generate(
                    prompt=dojo_prompt,
                    system_prompt=dojo_system_instruction,
                    model_name=model,
                )
                
                if not dojo_response:
                    print("Failed to generate Dojo response.")
                    break
                    
                user_utterance = dojo_response.strip()
                detailed_trace.append(f"User: {user_utterance}")
                dojo_transcript.append(f"User: {user_utterance}")
                
                if console_logging:
                    print(f"Dojo User: {user_utterance}")
                    
                if "<DOJO_GOAL_ACHIEVED>" in user_utterance or "<DOJO_GOAL_FAILED>" in user_utterance:
                    should_terminate = True
                    user_utterance = user_utterance.replace("<DOJO_GOAL_ACHIEVED>", "").replace("<DOJO_GOAL_FAILED>", "").strip()
                    
            except Exception as e:
                print(f"Error generating Dojo response: {e}")
                break
                
            turn += 1
            
        return detailed_trace

    def _generate_mermaid_graph(self, detailed_trace: List[str]) -> str:
        """Generate a Mermaid.js graph based on tool calls in the trace."""
        invoked_tools = set()
        for entry in detailed_trace:
            for line in entry.split("\n"):
                if "Tool Call:" in line or "Tool Call (Output):" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        tool_part = parts[1].split("with args")[0].strip()
                        invoked_tools.add(tool_part)
                        
        # Static graph for Bella Notte
        mermaid_str = """graph TD;
    Start([Start]) --> SZ[set_party_size];
    SZ --> DT[set_preferred_date];
    DT --> ST[set_selected_time];
    ST --> GN[set_guest_name];
    GN --> SR[set_special_requests];
    SR --> CP[confirm_pending];
"""
        
        # Highlight invoked tools
        for tool in invoked_tools:
            if "set_party_size" in tool: mermaid_str += "    style SZ fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            elif "set_preferred_date" in tool: mermaid_str += "    style DT fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            elif "set_selected_time" in tool: mermaid_str += "    style ST fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            elif "set_guest_name" in tool: mermaid_str += "    style GN fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            elif "set_special_requests" in tool: mermaid_str += "    style SR fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            elif "confirm_pending" in tool: mermaid_str += "    style CP fill:#d4edda,stroke:#27ae60,stroke-width:2px;\n"
            
        return mermaid_str


    def _extract_rules(self, doc_path: str, model: str) -> List[str]:
        """Extract a clean list of rules from instructions using Gemini."""
        with open(doc_path, "r") as f:
            instructions = f.read()
            
        prompt = f"""You are an advanced AI engineer auditing an agent's instructions.
Extract a clean, bulleted list of all operational rules, constraints, and protocols that the agent must follow.

Instructions:
{instructions}

Return only the list of rules, one per line, starting with a bullet point (-).
"""
        try:
            output = self.genai_client.generate(
                prompt=prompt,
                model_name=model,
            )
            if output:
                return [line.strip().removeprefix("- ").strip() for line in output.split("\n") if line.strip().startswith("-")]
        except Exception as e:
            print(f"Error extracting rules: {e}")
        return []




def get_app_name():
    from config import load_app_name as _load
    return _load()


def cmd_run_dynamic(args):
    """Run dynamic Dojo sim against live agent."""
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        app_name = f"projects/{cfg['gcp_project_id']}/locations/{cfg['location']}/apps/{cfg['deployed_app_id']}"
    else:
        app_name = get_app_name()

    model = args.model or _DEFAULT_MODEL
    sensei_model = args.sensei_model or "gemini-3.1-pro-preview"
    print(f"Running dynamic Dojo sim against {app_name}")
    
    sim = DojoSimRunner(app_name=app_name)
    
    # For single run, we just generate one persona directly or read from doc
    # Let's just read the doc and generate one persona for now as a simple test
    with open(args.doc, "r") as f:
        instructions = f.read()
        
    prompt = f"""Read the following agent instructions and generate a persona for a CUSTOMER / GUEST who is calling this agent.
The persona must have a specific goal (e.g., booking a table for 4) and a realistic background.
{instructions}
Return JSON matching {{ "persona": "...", "goal": "..." }}
"""
    try:
        class PersonaOutput(pydantic.BaseModel):
            persona: str
            goal: str
            
        output = sim.genai_client.generate(
            prompt=prompt,
            model_name=model,
            response_mime_type="application/json",
            response_schema=PersonaOutput,
        )
        if output:
            persona = output.persona
            goal = output.goal
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print(f"\nPersona: {persona}")
    print(f"Goal: {goal}")
    
    conv = sim.simulate_dynamic_conversation(
        persona_prompt=f"Persona: {persona}\nGoal: {goal}",
        model=model,
        console_logging=args.verbose,
    )
    
    # Sensei Evaluation
    print("\n=== Entering Sensei Vetting Phase ===")
    
    with open(args.doc, "r") as f:
        instructions = f.read()
        
    transcript_str = "\n".join(conv)
    
    sensei_prompt = f"""You are an advanced AI quality auditor (Sensei) evaluating a conversation between a user (Dojo Agent) and a restaurant host agent.
Your task is to evaluate whether the host agent followed the rules and protocol specified in the instructions.

Agent Instructions:
{instructions}

Conversation Transcript:
{transcript_str}

Evaluate the conversation and provide structured feedback. Identify any rule violations, incorrect tool calls, or poor conversational flow.
Return the result as a JSON object matching this schema:
{{
  "success": true/false, 
  "score": 0.0-100.0, 
  "findings": [
    {{
      "type": "Rule Violation" / "Ambiguity" / "Good Flow",
      "description": "Detail of what happened",
      "justification": "Why this is a finding based on instructions"
    }}
  ],
  "recommendations": ["List of actionable improvements for the agent prompt or instructions"]
}}
"""
    try:
        class SenseiOutput(pydantic.BaseModel):
            class Finding(pydantic.BaseModel):
                type: str
                description: str
                justification: str
            success: bool
            score: float
            findings: List[Finding]
            recommendations: List[str]
            
        output: SenseiOutput = sim.genai_client.generate(
            prompt=sensei_prompt,
            model_name=sensei_model,
            response_mime_type="application/json",
            response_schema=SenseiOutput,
        )
        
        if output:
            print(f"\nSensei Score: {output.score}/100.0")
            print(f"Success: {output.success}")
            print("\nFindings:")
            for f in output.findings:
                print(f"  - [{f.type}] {f.description}")
                print(f"    Justification: {f.justification}")
            print("\nRecommendations:")
            for r in output.recommendations:
                print(f"  - {r}")
                
            # Generate HTML report for this single run
            gym_results = [{
                "scenario_id": "DYNAMIC_RUN",
                "success": output.success,
                "score": output.score,
                "findings": [f.model_dump() for f in output.findings],
                "recommendations": output.recommendations
            }]
            
            doc_dir = os.path.dirname(args.doc)
            html_report_path = os.path.join(doc_dir, "dojo_report.html")
            
            coverage_data = [] 
            
            batch_transcripts = [{
                "scenario_id": "DYNAMIC_RUN",
                "transcript": transcript_str
            }]
            
            generate_gym_html_report(html_report_path, app_name, 1, 1 if output.success else 0, output.score, gym_results, coverage_data, batch_transcripts, sim)
            
    except Exception as e:
        print(f"Error in Sensei evaluation: {e}")

    print("\n--- Conversation Complete ---")



def cmd_run_gym(args):
    """Run full Dojo Gym batch simulations."""
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        app_name = f"projects/{cfg['gcp_project_id']}/locations/{cfg['location']}/apps/{cfg['deployed_app_id']}"
    else:
        app_name = get_app_name()

    model = args.model or _DEFAULT_MODEL
    sensei_model = args.sensei_model or "gemini-3.1-pro-preview"
    print(f"Entering Dojo Gym against {app_name}")
    
    sim = DojoSimRunner(app_name=app_name)
    matrix = sim._load_scenario_matrix(args.doc, model)
    
    if not matrix or "scenarios" not in matrix:
        print("Failed to load scenario matrix. Aborting.")
        return
        
    print(f"Loaded matrix with {len(matrix['scenarios'])} scenarios.")
    
    batch_transcripts = []
    
    for scenario in matrix["scenarios"]:
        print(f"\n[Gym Match] Running Scenario: {scenario['title']} ({scenario['type']})")
        print(f"Goal: {scenario['goal']}")
        
        persona_prompt = f"""Persona: You are a user testing the agent.
Scenario: {scenario['description']}
Goal: {scenario['goal']}
Hidden Agendas: {', '.join(scenario.get('hidden_agendas', []))}
"""
        
        print("Starting dynamic conversation...")
        conv = sim.simulate_dynamic_conversation(
            persona_prompt=persona_prompt,
            model=model,
            console_logging=args.verbose,
        )
        
        batch_transcripts.append({
            "scenario_id": scenario["id"],
            "transcript": "\n".join(conv)
        })
        
    print("\n=== Entering Sensei Vetting Phase ===")
    
    with open(args.doc, "r") as f:
        instructions = f.read()
        
    gym_results = []
    
    for item in batch_transcripts:
        scen_id = item["scenario_id"]
        transcript_str = item["transcript"]
        
        print(f"\nGrading Scenario {scen_id}...")
        
        sensei_prompt = f"""You are an advanced AI quality auditor (Sensei) evaluating a conversation between a user (Dojo Agent) and a restaurant host agent.
Your task is to evaluate whether the host agent followed the rules and protocol specified in the instructions.

Agent Instructions:
{instructions}

Conversation Transcript:
{transcript_str}

Evaluate the conversation and provide structured feedback. Identify any rule violations, incorrect tool calls, or poor conversational flow.
Return the result as a JSON object matching this schema:
{{
  "success": true/false, 
  "score": 0.0-100.0, 
  "findings": [
    {{
      "type": "Rule Violation" / "Ambiguity" / "Good Flow",
      "description": "Detail of what happened",
      "justification": "Why this is a finding based on instructions"
    }}
  ],
  "recommendations": ["List of actionable improvements for the agent prompt or instructions"]
}}
"""
        try:
            class SenseiOutput(pydantic.BaseModel):
                class Finding(pydantic.BaseModel):
                    type: str
                    description: str
                    justification: str
                success: bool
                score: float
                findings: List[Finding]
                recommendations: List[str]
                
            output: SenseiOutput = sim.genai_client.generate(
                prompt=sensei_prompt,
                model_name=sensei_model,
                response_mime_type="application/json",
                response_schema=SenseiOutput,
            )
            
            if output:
                gym_results.append({
                    "scenario_id": scen_id,
                    "success": output.success,
                    "score": output.score,
                    "findings": [f.model_dump() for f in output.findings],
                    "recommendations": output.recommendations
                })
                print(f"  Score: {output.score}/100.0 | Success: {output.success}")
                
        except Exception as e:
            print(f"Error in Sensei evaluation for {scen_id}: {e}")
            
    # Pass 3: Coverage Extraction
    print("\nExtracting rule coverage...")
    rules_list = sim._extract_rules(args.doc, model)
    
    coverage_prompt = f"""You are an advanced AI quality auditor evaluating test coverage.
Given the following list of operational rules and the batch of conversation transcripts, determine which rules were exercised in which scenarios, and whether the behavior was compliant.

Operational Rules:
{json.dumps(rules_list, indent=2)}

Batch Transcripts:
{json.dumps([{"id": r["scenario_id"], "text": r["transcript"]} for r in batch_transcripts], indent=2)}

Return a JSON object matching this schema:
{{
  "coverage": [
    {{
      "rule": "The exact rule text from the list",
      "scenarios_tested": [
        {{
          "scenario_id": "SCENARIO_01",
          "status": "Passed" / "Failed" / "Not Tested",
          "justification": "Brief reason"
        }}
      ]
    }}
  ]
}}
"""
    coverage_data = []
    try:
        class RuleCoverage(pydantic.BaseModel):
            class ScenarioTest(pydantic.BaseModel):
                scenario_id: str
                status: str
                justification: str
            rule: str
            scenarios_tested: List[ScenarioTest] = []
            
        class CoverageOutput(pydantic.BaseModel):
            coverage: List[RuleCoverage] = []
            
        coverage_output: CoverageOutput = sim.genai_client.generate(
            prompt=coverage_prompt,
            model_name=sensei_model,
            response_mime_type="application/json",
            response_schema=CoverageOutput,
        )
        
        if coverage_output:
            print("\nRule Coverage Extracted.")
            coverage_data = [c.model_dump() for c in coverage_output.coverage]
            
    except Exception as e:
        print(f"Error extracting coverage: {e}")

    print("\n" + "=" * 60)

    print("FINAL DOJO GYM REPORT")
    print("=" * 60)
    
    total_scenarios = len(batch_transcripts)
    passed_scenarios = sum(1 for r in gym_results if r["success"])
    avg_score = sum(r["score"] for r in gym_results) / len(gym_results) if gym_results else 0
    
    print(f"Passed Scenarios: {passed_scenarios}/{total_scenarios}")
    print(f"Average Compliance Score: {avg_score:.1f}/100.0")
    
    if passed_scenarios == total_scenarios:
        print("\n🏆 Congratulations! The agent has earned the Dojo Gym Badge!")
    else:
        print("\n❌ The agent failed to pass all challenges in the Dojo Gym.")
        
    doc_dir = os.path.dirname(args.doc)
    report_path = os.path.join(doc_dir, "dojo_gym_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "average_score": avg_score,
            "results": gym_results,
            "coverage": coverage_data
        }, f, indent=2)

        
    print(f"\nFull Gym report saved to: {report_path}")
    
    html_report_path = os.path.join(doc_dir, "dojo_gym_report.html")
    generate_gym_html_report(html_report_path, app_name, total_scenarios, passed_scenarios, avg_score, gym_results, coverage_data, batch_transcripts, sim)



def generate_gym_html_report(output_path, app_name, total_scenarios, passed_scenarios, avg_score, gym_results, coverage_data, batch_transcripts, sim):
    """Generate a rich HTML report for the Dojo Gym run with visualizations."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Dojo Gym Report - {ts}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true}});</script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
  h2 {{ color: #1a1a2e; margin-top: 30px; }}
  .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
  .summary .big {{ font-size: 2em; font-weight: bold; }}
  .pass {{ color: #27ae60; }} .fail {{ color: #e74c3c; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  th {{ background: #2c3e50; color: white; }}
  tr:hover {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
  .badge.pass {{ background: #d4edda; color: #155724; }}
  .badge.fail {{ background: #f8d7da; color: #721c24; }}
  .eval-card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; overflow: hidden; }}
  .eval-header {{ padding: 12px 16px; font-weight: bold; }}
  .eval-header.pass-bg {{ background: #d4edda; border-left: 4px solid #27ae60; }}
  .eval-header.fail-bg {{ background: #f8d7da; border-left: 4px solid #e74c3c; }}
  .eval-body {{ padding: 16px; }}
  .transcript {{ background: #f8f9fa; border-radius: 6px; padding: 12px; margin: 8px 0; font-size: 0.9em; white-space: pre-wrap; }}
</style>
</head><body>
<h1>Dojo Gym Evaluation Report</h1>
<div class="summary">
  <div class="big {('pass' if passed_scenarios == total_scenarios else 'fail')}">{passed_scenarios}/{total_scenarios} Passed</div>
  <div>Average Compliance Score: {avg_score:.1f}/100.0</div>
  <div style="margin-top:10px;">
    {('🏆 <b>Congratulations! The agent earned the Dojo Gym Badge!</b>' if passed_scenarios == total_scenarios else '❌ <b>The agent failed to pass all challenges.</b>')}
  </div>
</div>

<h2>PRD Rule Coverage</h2>
<table>
  <tr><th>Rule</th><th>Scenarios Tested</th><th>Status</th></tr>
"""
    
    for rule in coverage_data:
        html += f"  <tr><td>{rule['rule']}</td><td>"
        for st in rule.get('scenarios_tested', []):
            cls = "pass" if st['status'] == "Passed" else "fail" if st['status'] == "Failed" else "system"
            html += f"<span class='badge {cls}'>{st['scenario_id']} ({st['status']})</span> "
        html += "</td><td>"
        all_passed = all(st['status'] == "Passed" for st in rule.get('scenarios_tested', [])) if rule.get('scenarios_tested') else False
        if all_passed:
             html += "<span class='badge pass'>Covered</span>"
        elif any(st['status'] == "Failed" for st in rule.get('scenarios_tested', [])):
             html += "<span class='badge fail'>Failing</span>"
        else:
             html += "<span class='badge'>Not Tested</span>"
        html += "</td></tr>\n"
        
    html += "</table>\n\n<h2>Scenario Details & Call Graphs</h2>\n"
    
    for res in gym_results:
        scen_id = res['scenario_id']
        cls = "pass-bg" if res['success'] else "fail-bg"
        html += f'<div class="eval-card">\n'
        html += f'<div class="eval-header {cls}">{scen_id} — Score: {res["score"]}/100.0</div>\n'
        html += f'<div class="eval-body">\n'
        
        trans = next((t['transcript'] for t in batch_transcripts if t['scenario_id'] == scen_id), "")
        mermaid_str = sim._generate_mermaid_graph(trans.split("\n"))
        
        html += f'<h3>Call Graph Path</h3>\n'
        html += f'<div class="mermaid">\n{mermaid_str}\n</div>\n'
        
        html += f'<h3>Sensei Findings</h3>\n<ul>\n'
        for f in res.get('findings', []):
             html += f"<li><b>[{f['type']}]</b> {f['description']}<br><small>{f['justification']}</small></li>\n"
        html += "</ul>\n"
        
        html += f'<h3>Transcript</h3>\n<div class="transcript">{trans}</div>\n'
        html += '</div></div>\n'
        
    html += "</body></html>"
    
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML Gym report saved to: {output_path}")


def main():

    try:
        import cxas_scrapi  # noqa: F401
    except ImportError:
        print("Error: cxas-scrapi not installed. Activate venv (source .venv/bin/activate) and install cxas-scrapi first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Dojo & Sensei Simulation Runner (SCRAPI)")
    sub = parser.add_subparsers(dest="command")

    p_run_dynamic = sub.add_parser("run-dynamic", help="Run dynamic Dojo sim against live agent")
    p_run_dynamic.add_argument("--doc", required=True, help="Path to agent instructions (e.g. examples/bella_notte/agent_instruction.md)")
    p_run_dynamic.add_argument("--config", default=None, help="Path to a specific gecx-config.json file")
    p_run_dynamic.add_argument("--verbose", action="store_true")
    p_run_dynamic.add_argument("--model", default=None)
    p_run_dynamic.add_argument("--sensei-model", default=None, help="Model for Sensei evaluation (default: gemini-3.1-pro-preview)")

    p_run_gym = sub.add_parser("run-gym", help="Run full Dojo Gym batch simulations")
    p_run_gym.add_argument("--doc", required=True, help="Path to agent instructions (e.g. examples/bella_notte/agent_instruction.md)")
    p_run_gym.add_argument("--config", default=None, help="Path to a specific gecx-config.json file")
    p_run_gym.add_argument("--verbose", action="store_true")
    p_run_gym.add_argument("--model", default=None)
    p_run_gym.add_argument("--sensei-model", default=None, help="Model for Sensei evaluation (default: gemini-3.1-pro-preview)")

    args = parser.parse_args()
    commands = {"run-dynamic": cmd_run_dynamic, "run-gym": cmd_run_gym}

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
