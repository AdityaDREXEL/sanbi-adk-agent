"""Quick logic tests for the growth port (no network, no LLM)."""
import sys
sys.path.insert(0, ".")

from sanbi_core.platforms import classify_url
from sanbi_core.growth import build_opportunities, get_playbook

checks = [
    ("https://www.reddit.com/r/lasik/comments/abc123/my_experience", "reddit"),
    ("https://forum.digikey.com/t/current-sensor/123", "forum"),
    ("https://www.allaboutvision.com/lasik/cost/", None),
    ("https://www.allaboutvision.com/blog/lasik-cost/", "blog"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://www.yelp.com/biz/sight360-philadelphia", "reviews"),
    ("https://docs.python.org/3/", None),
    ("https://electronics.stackexchange.com/questions/1", "stackexchange"),
]
for url, want in checks:
    d, got = classify_url(url)
    assert got == want, f"{url}: want {want}, got {got}"
print("classifier: 8/8 OK")

log = [
    {"prompt": "best lasik philly", "engine": "gemini",
     "grade": {"cited_sources": ["https://reddit.com/r/lasik/comments/x1/thread",
                                 "https://blog.example.com/lasik-guide"], "source_titles": {}}},
    {"prompt": "best lasik philly", "engine": "openai",
     "grade": {"cited_sources": ["https://reddit.com/r/lasik/comments/x1/thread"], "source_titles": {}}},
    {"prompt": "lasik cost", "engine": "gemini",
     "grade": {"cited_sources": ["https://reddit.com/r/lasik/comments/x1/thread",
                                 "https://yelp.com/biz/sight360"], "source_titles": {}}},
]
opps = build_opportunities(log)
top = opps[0]
assert top["platform"] == "reddit" and top["source_url"].endswith("/thread")
# 2 engines, 2 prompts, recency 1.0 -> (50+10+20)*1.30 = 104.0
assert top["score"] == 104.0, top["score"]
assert get_playbook("reviews")["action_type"] == "review_acquisition"
assert get_playbook("blog")["action_type"] == "counter_content"
print("materializer OK:", [(o["platform"], o["score"]) for o in opps])
print("ALL TESTS PASS")
