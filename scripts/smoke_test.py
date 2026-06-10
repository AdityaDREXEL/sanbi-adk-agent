"""
scripts/smoke_test.py — Verify Vertex AI + OpenAI clients before building further.

Run after configuring .env:
    python scripts/smoke_test.py
"""

import asyncio
import sys
from pathlib import Path

# Allow `python scripts/smoke_test.py` from the repo root (and anywhere else).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> int:
    ok = True

    # 1. Vertex Gemini
    print("— Vertex Gemini —")
    try:
        from sanbi_core.llm import gemini_client, GEMINI_MODEL
        if not gemini_client:
            print("  ❌ gemini_client is None — check GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT")
            ok = False
        else:
            resp = await gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents="what is 2+2? answer with just the number"
            )
            print(f"  ✅ {GEMINI_MODEL} → {resp.text.strip()[:40]}")
    except Exception as e:
        print(f"  ❌ {e}")
        ok = False

    # 2. Grounded search (the identity pipeline depends on this)
    print("— Gemini + Google Search grounding —")
    try:
        from sanbi_core.gemini import gemini_grounded_text
        text, sources = await gemini_grounded_text("Who founded Allegro MicroSystems?", use_search=True)
        if not text:
            print("  ❌ empty grounded response — client not initialized or call failed")
            ok = False
        else:
            print(f"  ✅ {len(text)} chars, {len(sources)} grounded sources")
    except Exception as e:
        print(f"  ❌ {e}")
        ok = False

    # 3. OpenAI
    print("— OpenAI —")
    try:
        from sanbi_core.llm import openai_client, OPENAI_MODEL
        if not openai_client:
            print("  ❌ openai_client is None — check OPENAI_API_KEY")
            ok = False
        else:
            resp = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": "say ok"}],
            )
            print(f"  ✅ {OPENAI_MODEL} → {resp.choices[0].message.content.strip()[:40]}")
    except Exception as e:
        print(f"  ❌ {e}")
        ok = False

    print("\n" + ("✅ ALL SYSTEMS GO" if ok else "❌ FIX THE ABOVE BEFORE PROCEEDING"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
