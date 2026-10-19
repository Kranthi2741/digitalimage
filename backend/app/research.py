"""Research-only trial workflow for evidence-backed social content."""
import json
import re
import os
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse
from app.llm import _text_generate

ALLOWED_DOMAINS = ("cdc.gov", "who.int", "nih.gov", "nhs.uk", "gov.uk", "mayoclinic.org")

QUESTION_TERMS = {
    "What is it?": ("defined", "refers to", "means", "definition", "mortality", "loss"),
    "Why does it happen?": ("cause", "because", "risk factor", "contribut", "prematur", "infection", "condition"),
    "Who is at risk?": ("risk", "higher", "population", "women", "infant", "mother", "pregnant"),
    "What are the signs?": ("sign", "symptom", "warning", "recognize", "fever", "bleeding", "breath"),
    "How is it assessed?": ("assess", "diagnos", "screen", "test", "monitor", "review", "surveillance"),
    "How is it treated or managed?": ("treat", "manage", "care", "intervention", "clinical", "hospital"),
    "How can risk be reduced?": ("prevent", "reduce", "prenatal", "vaccin", "safe sleep", "avoid", "health care"),
    "When should urgent help be sought?": ("urgent", "emergency", "seek", "warning", "immediate", "call", "danger"),
}


def _clean_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]|\([^)]*https?://[^)]*\)", " ", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\s+", " ", unescape(text)).strip()
    raw = re.split(r"(?<=[.!?])\s+|\s+\|\s+", text)
    output = []
    for sentence in raw:
        sentence = sentence.strip(" -:;,.[]")
        if len(sentence) < 45 or len(sentence.split()) < 7 or len(sentence) > 420:
            continue
        lowered = sentence.casefold()
        if any(noise in lowered for noise in ("websites are changing", "follow us", "cookie", "javascript", "assistive technology")):
            continue
        normalized = re.sub(r"[^a-z0-9 ]", "", lowered)
        if normalized and normalized not in {re.sub(r"[^a-z0-9 ]", "", x.casefold()) for x in output}:
            output.append(sentence)
    return output


def _relevant_excerpt(text: str, question: str, limit: int = 2) -> str:
    sentences = _clean_sentences(text)
    terms = QUESTION_TERMS.get(question, ())
    ranked = sorted(sentences, key=lambda s: sum(s.casefold().count(term) for term in terms), reverse=True)
    selected = [sentence for sentence in ranked if any(term in sentence.casefold() for term in terms)][:limit]
    return " ".join(selected)

# Direct official pages keep the trial usable when a search provider changes its
# HTML or is unavailable. These are source candidates, not generated claims.
class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            text = re.sub(r"\s+", " ", unescape(data)).strip()
            if text:
                self.parts.append(text)


def _page_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "MarketAI research trial/1.0"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(400000).decode("utf-8", errors="replace")
    parser = _TextParser()
    parser.feed(raw)
    return " ".join(parser.parts)[:12000]


def run_research(prompt: str) -> dict:
    if not prompt or len(prompt.strip()) < 8:
        raise ValueError("Please enter a specific research topic or question.")
    instruction = f"""
Research this user request from your available knowledge: {prompt}

Create a concise 4-5 image educational carousel plan, not a long report.
Use the questions that fit the topic: what it is, why it happens, who is at risk,
signs or warning signs, how professionals assess it, treatment or management,
prevention/risk reduction, and when urgent help is needed. Do not force irrelevant
sections. For simple occasions such as Diwali or Doctors Day, return campaign_mode
single_image with one image instead of a carousel.

For medical topics, use accurate wording, never promise a cure, separate maternal
mortality, infant mortality, pregnancy loss, and stillbirth unless the request joins
them, and mark medical_review_required true.

Develop one distinctive art_direction for the whole campaign: palette, typography,
illustration/photography treatment and emotional tone appropriate to this request.
Each slide needs its own specific visual concept and composition. Integrate
meaningful icons and explanatory illustrations with imagery and typography;
do not prescribe the same photograph-above-footer format on every slide.
Preserve any explicit doctor/patient interaction requested by the user.
Screening eligibility depends on the person and cancer type; avoid blanket
screening promises and distinguish screening from evaluation of symptoms.

Return ONLY valid JSON in this shape. Do not put URLs, citations, source names, or reference numbers inside visible headlines, answers, or short_copy; sources are stored internally by the application:
{{"topic":"","campaign_mode":"single_image or educational_carousel","summary":"","art_direction":"",
"research_sections":[{{"question":"","answer":"","source_numbers":[1]}}],
"key_facts":[{{"claim":"","source_numbers":[1],"confidence":"high or review"}}],
"slide_outline":[{{"number":1,"purpose":"","headline":"","short_copy":"","bullet_points":["","",""],"visual_direction":"","source_numbers":[1]}}],
"cautions":[""],"questions_for_review":[""],"disclaimer":"",
"medical_review_required":true}}
Keep the response compact: summary under 45 words, no more than 4 research_sections, no more than 5 key_facts, and no more than 5 slides. Keep each slide's short_copy under 25 words and each bullet_point under 9 words. Use 2-4 short bullet_points directly related to the visual, use plain language, and make all text suitable for an infographic image. Include source numbers only in source_numbers fields. Never add commentary outside the JSON.
"""
    try:
        raw = (_text_generate(instruction + "\nDo not omit any required key. Populate research_sections, key_facts, slide_outline, cautions, summary, disclaimer, and medical_review_required with useful content.") or "").strip()
        if not raw:
            raise RuntimeError("Qwen3-8B returned no content")
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Qwen3-8B returned non-JSON content: {raw[:300]}") from error
    except Exception as error:
        raise RuntimeError(f"Qwen3-8B research failed: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("Qwen3-8B returned an invalid research response.")
    required = ("summary", "research_sections", "key_facts", "slide_outline", "cautions", "disclaimer")
    missing = [key for key in required if key not in result or result.get(key) in (None, "")]
    if missing:
        # Smaller local models often stop after the metadata fields when given
        # the full carousel schema. Retry with only the essential structure.
        compact = f'''Return ONLY valid JSON for this research topic: {prompt}
Use exactly these keys: summary (string), research_sections (array of objects with question and answer),
key_facts (array of objects with claim and confidence), slide_outline (array of objects with number, headline, short_copy, purpose),
cautions (array of strings), disclaimer (string), medical_review_required (boolean).
Provide useful concise educational content. Do not omit keys.'''
        try:
            compact_raw = (_text_generate(compact) or "").strip()
            if compact_raw.startswith("```"):
                compact_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", compact_raw, flags=re.IGNORECASE | re.DOTALL).strip()
            start, end = compact_raw.find("{"), compact_raw.rfind("}")
            compact_result = json.loads(compact_raw[start:end + 1] if start >= 0 and end > start else compact_raw)
            if isinstance(compact_result, dict):
                result.update(compact_result)
        except Exception:
            pass
        missing = [key for key in required if key not in result or result.get(key) in (None, "")]
        if missing:
            raise RuntimeError(f"Qwen3-8B returned an incomplete research response; missing: {', '.join(missing)}")
    result["sources"] = []
    result["research_mode"] = f"Local Hugging Face {os.environ.get('HF_TEXT_MODEL', 'Qwen/Qwen3-8B')}; human review required"
    result["user_prompt"] = prompt
    result["medical_review_required"] = bool(result.get("medical_review_required", True))
    return result
