"""
OpenEnv-Orbital-Command | agent/llm_agent.py

ReAct-style LLM orchestrator.
Supports: Google Gemini, HF Inference, and a built-in RuleBased fallback.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on the path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional

from env.models import Action, ActionType, Observation
from env.physics import (has_line_of_sight, is_in_sunlight,
                          steps_until_eclipse, steps_until_sunlight,
                          sat_positions_to_isl_graph, find_min_hop_path)
from agent.prompt_builder import build_observation_prompt, build_system_prompt


# ---------------------------------------------------------------------------
# Base agent interface
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract base for all agent backends."""

    def __init__(self, name: str = "Agent"):
        self.name = name
        self._step_history: List[Dict[str, Any]] = []
        self._call_count = 0
        self._total_latency_ms = 0.0

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def record_step(self, step: int, action: Action, reward: float):
        self._step_history.append({
            "step": step,
            "action": action.model_dump(),
            "reward": reward,
        })

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._total_latency_ms / self._call_count

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "agent_name":       self.name,
            "total_calls":      self._call_count,
            "avg_latency_ms":   round(self.avg_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Google Gemini agent
# ---------------------------------------------------------------------------

class GeminiAgent(BaseAgent):
    """
    Google Gemini API agent.
    Supports both SDK flavours:
      - google-generativeai   (pip install google-generativeai)
      - google-genai          (pip install google-genai)

    Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or environment.
    Get your key free at: https://aistudio.google.com/app/apikey
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        name: str = "Gemini",
    ):
        super().__init__(name=name)
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._client     = None   # lazily initialised
        self._sdk        = None   # "new" | "legacy"

    def _get_api_key(self) -> str:
        key = (os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("GEMINI_API_KEY", ""))
        if not key:
            raise EnvironmentError(
                "No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "in .env or environment.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        return key

    def _init_client(self):
        """Try new SDK first, fall back to legacy SDK."""
        api_key = self._get_api_key()

        # Try new SDK: google-genai
        try:
            from google import genai as _google_genai
            self._client = _google_genai.Client(api_key=api_key)
            self._sdk = "new"
            return
        except ImportError:
            pass

        # Fall back to legacy SDK: google-generativeai
        try:
            import google.generativeai as _genai
            _genai.configure(api_key=api_key)
            self._client = _genai.GenerativeModel(
                model_name=self.model,
                system_instruction=build_system_prompt(),
                generation_config=_genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            self._sdk = "legacy"
            return
        except ImportError:
            pass

        raise ImportError(
            "No Gemini SDK found. Install one of:\n"
            "  pip install google-genai          (recommended — supports Gemini 2.x)\n"
            "  pip install google-generativeai   (legacy)"
        )

    def act(self, obs: Observation) -> Action:
        if self._client is None:
            self._init_client()

        t0     = time.time()
        system = build_system_prompt()
        prompt = build_observation_prompt(obs, self._step_history)
        full_prompt = f"{system}\n\n{prompt}"   # Gemini takes system+user as single string

        try:
            if self._sdk == "new":
                from google.genai import types as _gtypes
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=_gtypes.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        system_instruction=system,
                    ),
                )
                raw = response.text or ""
            else:
                # Legacy SDK: GenerativeModel already has system_instruction baked in
                response = self._client.generate_content(prompt)
                raw = response.text or ""
        except Exception as ex:
            # Graceful fallback on quota / network errors
            return Action(
                action_type=ActionType.SLEEP_MODE,
                target_sat_id=obs.satellites[0].sat_id if obs.satellites else "unknown",
                reasoning=f"Gemini API error — sleeping. Details: {ex}",
            )

        self._call_count += 1
        self._total_latency_ms += (time.time() - t0) * 1000
        return _parse_action(raw)


# ---------------------------------------------------------------------------
# HuggingFace Inference agent  (kept below Gemini for ordering)
# ---------------------------------------------------------------------------

class HuggingFaceAgent(BaseAgent):
    """HuggingFace Inference Endpoint agent."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        name: str = "HuggingFace",
    ):
        super().__init__(name=name)
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=self.model,
                    token=os.environ.get("HF_TOKEN", ""),
                )
            except ImportError:
                raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
        return self._client

    def act(self, obs: Observation) -> Action:
        t0 = time.time()
        prompt = build_observation_prompt(obs, self._step_history)
        system = build_system_prompt()

        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result = client.chat_completion(messages=messages, max_tokens=512, temperature=0.3)
        raw = result.choices[0].message.content or ""
        self._call_count += 1
        self._total_latency_ms += (time.time() - t0) * 1000

        return _parse_action(raw)


# ---------------------------------------------------------------------------
# Rule-based fallback agent (no API key required)
# ---------------------------------------------------------------------------

class RuleBasedAgent(BaseAgent):
    """
    Deterministic heuristic agent — zero external dependencies.
    Priority order:
      1. Emergency thermal vent if satellite overheating.
      2. Sleep if battery < 25% and in sunlight.
      3. Downlink if satellite has data and LoS to online station.
      4. Capture if in range of pending request.
      5. ISL relay if data is trapped with no LoS.
      6. Sleep (default: conserve battery).
    """

    def __init__(self):
        super().__init__(name="RuleBasedAgent")

    def act(self, obs: Observation) -> Action:
        t0 = time.time()
        action = self._decide(obs)
        self._call_count += 1
        self._total_latency_ms += (time.time() - t0) * 1000
        return action

    def _decide(self, obs: Observation) -> Action:
        # Build lookup maps
        sat_map = {s.sat_id: s for s in obs.satellites}
        stn_map = {s.station_id: s for s in obs.ground_stations}
        req_map = {r.id: r for r in obs.imaging_requests}
        sat_positions = {s.sat_id: s.orbital_position for s in obs.satellites}
        isl_graph = obs.isl_topology

        # Priority 1: Thermal emergency
        for sat in sorted(obs.satellites, key=lambda s: -s.thermal_level):
            if sat.mode.value == "dead":
                continue
            if sat.thermal_level >= 75.0:
                return Action(
                    action_type=ActionType.THERMAL_VENT,
                    target_sat_id=sat.sat_id,
                    reasoning=f"THERMAL EMERGENCY: {sat.sat_id} at {sat.thermal_level:.0f}°, venting heat",
                )

        # Priority 2: Battery critical — sleep if in sunlight to charge
        for sat in sorted(obs.satellites, key=lambda s: s.battery_level):
            if sat.mode.value == "dead":
                continue
            if sat.battery_level < 25.0 and sat.in_sunlight:
                return Action(
                    action_type=ActionType.SLEEP_MODE,
                    target_sat_id=sat.sat_id,
                    reasoning=f"BATTERY CRITICAL: {sat.sat_id} at {sat.battery_level:.0f}%, entering sleep to charge",
                )

        # Priority 3: Downlink if data pending and in LoS
        for sat in obs.satellites:
            if sat.mode.value == "dead" or sat.pending_data_gb <= 0:
                continue
            if sat.line_of_sight_to_ground:
                stn = stn_map.get(sat.line_of_sight_to_ground)
                if stn and stn.status == "online":
                    return Action(
                        action_type=ActionType.DOWNLINK_DATA,
                        target_sat_id=sat.sat_id,
                        target_station=stn.station_id,
                        reasoning=f"DOWNLINK OPPORTUNITY: {sat.sat_id} has {sat.pending_data_gb:.2f} GB, LoS to {stn.station_id}",
                    )

        # Priority 4: Capture image if in range and battery sufficient
        for req in sorted(obs.imaging_requests,
                           key=lambda r: {"EMERGENCY": 0, "URGENT": 1, "ROUTINE": 2}[r.priority]):
            for sat in obs.satellites:
                if sat.mode.value == "dead":
                    continue
                if sat.battery_level < 15.0:
                    continue
                if sat.storage_used > 90.0:
                    continue
                if has_line_of_sight(sat.orbital_position, req.target_deg, fov_deg=15):
                    return Action(
                        action_type=ActionType.CAPTURE_IMAGE,
                        target_sat_id=sat.sat_id,
                        request_id=req.id,
                        reasoning=f"IN RANGE: {sat.sat_id} at {sat.orbital_position}° capturing {req.id} at {req.target_deg}°",
                    )

        # Priority 5: ISL relay if isolated data exists
        for sat in obs.satellites:
            if sat.mode.value == "dead" or sat.pending_data_gb <= 0:
                continue
            if sat.line_of_sight_to_ground:
                continue  # Already has direct downlink option
            # Try to find relay path to any online station
            for stn in obs.ground_stations:
                if stn.status != "online":
                    continue
                path = find_min_hop_path(isl_graph, sat.sat_id, stn.position_deg, sat_positions, stn.fov_deg)
                if path and len(path) > 1:
                    # Check relay batteries
                    relay_ok = all(
                        sat_map.get(hop, sat).battery_level >= 12.0
                        for hop in path
                    )
                    if relay_ok:
                        return Action(
                            action_type=ActionType.INTER_SATELLITE_LINK,
                            target_sat_id=sat.sat_id,
                            relay_chain=path,
                            reasoning=f"ISL RELAY: {' → '.join(path)} to reach {stn.station_id}",
                        )

        # Default: Sleep the most battery-depleted non-dead satellite
        candidates = [s for s in obs.satellites if s.mode.value != "dead"]
        if candidates:
            worst = min(candidates, key=lambda s: s.battery_level)
            return Action(
                action_type=ActionType.SLEEP_MODE,
                target_sat_id=worst.sat_id,
                reasoning=f"DEFAULT SLEEP: {worst.sat_id} at {worst.battery_level:.0f}% battery, conserving charge",
            )

        # Shouldn't reach here
        return Action(
            action_type=ActionType.SLEEP_MODE,
            target_sat_id=obs.satellites[0].sat_id if obs.satellites else "unknown",
            reasoning="Fallback: no clear decision — sleeping to conserve resources",
        )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agent(backend: str = "rule_based", **kwargs) -> BaseAgent:
    """
    Factory function to create an agent by backend name.

    Args:
        backend: "gemini" | "huggingface" | "rule_based"
        **kwargs: passed to the agent constructor
    """
    backends = {
        "gemini":       GeminiAgent,
        "huggingface":  HuggingFaceAgent,
        "rule_based":   RuleBasedAgent,
    }
    cls = backends.get(backend.lower())
    if cls is None:
        raise ValueError(f"Unknown backend '{backend}'. Options: {list(backends)}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> Action:
    """Extract JSON action from LLM response text."""
    # Try to find JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw_json = json_match.group(1)
    else:
        # Find first { ... } in the response
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        raw_json = brace_match.group(0) if brace_match else "{}"

    try:
        data = json.loads(raw_json)
        return Action(**data)
    except Exception as e:
        # Fallback: safe sleep action
        return Action(
            action_type=ActionType.SLEEP_MODE,
            target_sat_id="Sat-Alpha",
            reasoning=f"PARSE ERROR — defaulting to sleep. Raw: {raw[:200]}",
        )
