"""Phase 4 RAG logic tests."""
import sys, types, json

# ── mock deps ────────────────────────────────────────────────────────────
loguru_mod = types.ModuleType('loguru')
class _L:
    def debug(self,*a,**k): pass
    def info(self,*a,**k): pass
    def warning(self,*a,**k): pass
    def error(self,*a,**k): pass
loguru_mod.logger = _L()
sys.modules['loguru'] = loguru_mod

config_mod = types.ModuleType('config'); config_mod.__path__ = []
settings_mod = types.ModuleType('config.settings')
class S:
    GROQ_MODEL='llama-3.3-70b-versatile'; GEMINI_MODEL='gemini-1.5-flash'
    GROQ_API_KEY='test'; GOOGLE_API_KEY='test'
    LLM_MAX_TOKENS=4096; LLM_TEMPERATURE=0.1
    MAX_RAG_HOPS=3; QUALITY_GATE_THRESHOLD=0.70
    TOP_K_RETRIEVAL=10; METADATA_FILTER_ENABLED=True
    SIGNAL_LONG_THRESHOLD=0.30; SIGNAL_SHORT_THRESHOLD=-0.30
settings_mod.settings = S()
sys.modules['config'] = config_mod
sys.modules['config.settings'] = settings_mod
sys.path.insert(0, '.')

# ═══════════════════════════════════════════════════════
# TEST 1: _parse_json_response with various formats
# ═══════════════════════════════════════════════════════
from src.rag.llm_client import _parse_json_response, _is_rate_limit_error, _is_retryable

# Pure JSON object
r = _parse_json_response('{"score": 0.9, "issues": []}')
assert r == {'score': 0.9, 'issues': []}, f'Got {r}'
print('  pure JSON object ✓')

# JSON wrapped in ```json fences
r2 = _parse_json_response('```json\n{"score": 0.8}\n```')
assert r2 == {'score': 0.8}, f'Got {r2}'
print('  ```json fence ✓')

# JSON array wrapped in fences → wrapped in result key
r3 = _parse_json_response('```\n[{"metric": "EPS"}]\n```')
assert r3 == {'result': [{'metric': 'EPS'}]}, f'Got {r3}'
print('  array fence ✓')

# JSON embedded in surrounding text
r4 = _parse_json_response('Some text {"key": "val"} after')
assert r4 == {'key': 'val'}, f'Got {r4}'
print('  embedded JSON ✓')

# Non-JSON
r5 = _parse_json_response('not json at all')
assert r5 == {}, f'Got {r5}'
print('  non-JSON → empty dict ✓')

print('_parse_json_response: all 5 cases ✓')

# Rate limit detection
class FakeExc(Exception): pass
assert _is_rate_limit_error(FakeExc('rate limit exceeded'))
assert _is_rate_limit_error(FakeExc('429 too many requests'))
assert not _is_rate_limit_error(FakeExc('connection error'))
print('_is_rate_limit_error ✓')

assert _is_retryable(FakeExc('timeout'))
assert _is_retryable(FakeExc('502 bad gateway'))
assert not _is_retryable(FakeExc('invalid api key'))
print('_is_retryable ✓')

# ═══════════════════════════════════════════════════════
# TEST 2: prompts
# ═══════════════════════════════════════════════════════
from src.rag.prompts import (
    build_synthesis_prompt, build_gap_detection_prompt,
    build_quality_check_prompt, format_contradictions,
    SYSTEM_EARNINGS_ANALYST,
)

p = build_synthesis_prompt(
    ticker='AAPL', period='Q3 2024', context='Sample context',
    quality_score=0.45, signal='LONG',
    sentiment_drift_component=0.60, guidance_accuracy_component=0.50,
    accruals_component=0.46, analyst_revision_component=0.65,
    alert_level='GREEN', alert_reason='Stable',
    contradiction_count=0, contradiction_summary='',
)
assert 'AAPL' in p and 'LONG' in p and '+0.600' in p
print(f'synthesis_prompt: {len(p)} chars ✓')

gap_p = build_gap_detection_prompt('AAPL', 'How has revenue trended?', 'Q3 context')
assert 'has_gaps' in gap_p and 'AAPL' in gap_p
print(f'gap_detection_prompt: {len(gap_p)} chars ✓')

contras = [{'contradiction_score': 0.87, 'chunk_a_source': 'A', 'chunk_b_source': 'B', 'interpretation': 'Test'}]
fmt = format_contradictions(contras)
assert '\u26a0\ufe0f CONFLICT' in fmt
print(f'format_contradictions: {len(fmt)} chars ✓')

assert 'EarningsEdge' in SYSTEM_EARNINGS_ANALYST
print('SYSTEM_EARNINGS_ANALYST ✓')

# ═══════════════════════════════════════════════════════
# TEST 3: nodes helpers
# ═══════════════════════════════════════════════════════
from src.rag.nodes import _build_metadata_filter, _format_context, _get_peer_tickers

f1 = _build_metadata_filter('AAPL', 'guidance_tracking')
assert f1['ticker'] == 'AAPL' and 'section_type' in f1
print(f'_build_metadata_filter(guidance_tracking) ✓')

f2 = _build_metadata_filter('AAPL', 'general')
assert 'section_type' not in f2
print('_build_metadata_filter(general): no section filter ✓')

chunks = [{'chunk_id':'c1','text':'Revenue grew 8%','ticker':'AAPL',
           'filing_type':'10-Q','section_type':'guidance','quarter':'Q3','year':2024,'filed_date':'2024-08-02'}]
ctx = _format_context(chunks)
assert 'AAPL' in ctx and 'SOURCE 1' in ctx
print(f'_format_context: {len(ctx)} chars ✓')

peers = _get_peer_tickers('AAPL')
assert 'MSFT' in peers
print(f'_get_peer_tickers(AAPL): {peers} ✓')

# ═══════════════════════════════════════════════════════
# TEST 4: graph routing
# ═══════════════════════════════════════════════════════
from src.rag.graph import _route_after_gap_check, _route_after_quality_check

assert _route_after_gap_check({'needs_more_retrieval': True,  'hop_count': 1}) == 'industry_retrieval'
assert _route_after_gap_check({'needs_more_retrieval': False, 'hop_count': 1}) == 'contradiction_check'
assert _route_after_gap_check({'needs_more_retrieval': True,  'hop_count': 99}) == 'contradiction_check'
print('_route_after_gap_check: all 3 cases ✓')

assert _route_after_quality_check({'quality_score': 0.40, 'hop_count': 1}) == 'company_retrieval'
assert _route_after_quality_check({'quality_score': 0.90, 'hop_count': 1}) == 'ragas_prep'
assert _route_after_quality_check({'quality_score': 0.20, 'hop_count': 99}) == 'ragas_prep'
print('_route_after_quality_check: all 3 cases ✓')

# ═══════════════════════════════════════════════════════
# TEST 5: EarningsEdgeResult
# ═══════════════════════════════════════════════════════
from src.rag.multi_hop_chain import EarningsEdgeResult

r = EarningsEdgeResult(
    ticker='AAPL', query='test', quarter='Q3', year=2024,
    quality_score_obj={'composite_score': 0.45},
    tone_drift_report={'alert_level': 'RED'},
)
assert r.composite_score == 0.45
assert r.signal == 'LONG'
assert r.alert_level == 'RED'
print(f'EarningsEdgeResult(LONG): composite={r.composite_score}  signal={r.signal}  alert={r.alert_level} ✓')

r2 = EarningsEdgeResult(
    ticker='XYZ', query='test', quarter='Q1', year=2024,
    quality_score_obj={'composite_score': -0.55},
    tone_drift_report={},
)
assert r2.signal == 'SHORT'
assert r2.alert_level == 'GREEN'   # default
print(f'EarningsEdgeResult(SHORT): signal={r2.signal}  alert={r2.alert_level} ✓')

print()
print('All Phase 4 RAG logic tests PASSED \u2713')
