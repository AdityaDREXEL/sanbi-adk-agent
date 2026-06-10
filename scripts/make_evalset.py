"""Generate the routing evalset using ADK's own pydantic models (schema-guaranteed).

Regenerate with:  python scripts/make_evalset.py
Run the eval (needs LLM credentials):
    adk eval agents/sanbi_audit agents/sanbi_audit/evalsets/routing.evalset.json \
        --config_file_path agents/sanbi_audit/evalsets/test_config.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_set import EvalSet

OUT = Path(__file__).resolve().parent.parent / "agents" / "sanbi_audit" / "evalsets"
OUT.mkdir(parents=True, exist_ok=True)

now = time.time()

evalset = EvalSet(
    eval_set_id="sanbi_routing",
    name="Sanbi coordinator routing + capability framing",
    description=(
        "Checks the coordinator explains the measure->act pipeline without "
        "invoking tools, and asks for the inputs it needs (domain + topic). "
        "Cheap to run: no audit is executed."
    ),
    eval_cases=[
        EvalCase(
            eval_id="capabilities_pitch",
            conversation=[
                Invocation(
                    invocation_id="inv_capabilities_1",
                    user_content=types.Content(
                        role="user",
                        parts=[types.Part(text="What can you do?")],
                    ),
                    final_response=types.Content(
                        role="model",
                        parts=[types.Part(text=(
                            "I measure a brand's visibility inside AI assistants — "
                            "I research the brand, audit real buyer queries across "
                            "OpenAI and Gemini, and grade who gets recommended. Then "
                            "I act on the findings: I rank and verify the sources AI "
                            "engines cite and draft growth actions for each platform. "
                            "Share a brand domain and topic to start an audit."
                        ))],
                    ),
                    creation_timestamp=now,
                ),
            ],
            creation_timestamp=now,
        ),
    ],
    creation_timestamp=now,
)

path = OUT / "routing.evalset.json"
path.write_text(evalset.model_dump_json(indent=2, exclude_none=True))
print(f"wrote {path}")

config = {"criteria": {"response_match_score": 0.35}}
cfg_path = OUT / "test_config.json"
cfg_path.write_text(json.dumps(config, indent=2))
print(f"wrote {cfg_path}")
