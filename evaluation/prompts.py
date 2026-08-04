"""evaluation/prompts.py — Single source of truth for all evaluation prompts.

This module centralizes every prompt string used across the SIGNPOST-Bench
evaluation pipeline and paper figure generation:

  - COORDINATE_PROMPT         — standard geo-localization (api_client + paper figure)
  - PROBING_PROMPT            — structured conflict probing (evaluate_probing + paper figure)
  - DEFENSE_PROMPT            — conflict-aware defense prompting (evaluate_probing + paper figure)
  - build_attack_generation_prompt / ATTACK_GENERATION_PROMPT
                              - shared scene-text replacement generation
  - GENERALIZATION_TASKS      — cross-task generalization prompts (evaluate_generalization + paper figure)

Consumers should import from here instead of duplicating the text, so that the
paper's prompt-card figures always match the prompts actually sent to models.
"""

# =============================================================================
#  Coordinate prediction (standard evaluation)
# =============================================================================

COORDINATE_PROMPT = """Analyze this photo and determine where it was taken.
You MUST provide your best estimate of GPS coordinates even if uncertain.
Do NOT refuse. Always give a coordinate guess.
Output ONLY: (Latitude, Longitude)
Example: (48.8584, 2.2945)"""

# =============================================================================
#  Structured conflict probing
# =============================================================================

PROBING_PROMPT = """Analyze this photo carefully to determine where it was taken.

You must complete ALL of the following steps:

Step 1 — VISUAL EVIDENCE: List the key visual cues that suggest the location (e.g., architecture style, vegetation, road layout, terrain, traffic conventions, climate indicators, vehicle types).

Step 2 — TEXTUAL EVIDENCE: List any readable text in the image (signs, storefronts, addresses, license plates) and what geographic location each piece of text suggests.

Step 3 — CONSISTENCY CHECK: Decide whether the visual evidence and textual evidence point to the SAME geographic region. Answer true or false.

Step 4 — EVIDENCE TRUST: Which type of evidence do you trust MORE for determining the actual location: "Visual", "Textual", or "Both"? Briefly explain why.

Step 5 — FINAL PREDICTION: Based on your analysis, provide your best GPS coordinates.

You MUST output ONLY valid JSON in exactly this format:
{
  "visual_evidence": ["cue1", "cue2"],
  "visual_region_estimate": "estimated region from visual cues only",
  "textual_evidence": ["text1 -> suggests LocationA"],
  "textual_region_estimate": "estimated region from text only",
  "consistent": true,
  "consistency_explanation": "brief explanation",
  "trusted_source": "Visual",
  "trust_explanation": "brief explanation",
  "final_prediction": "(latitude, longitude)"
}"""

# =============================================================================
#  Conflict-aware defense prompting
# =============================================================================

DEFENSE_PROMPT = """Analyze this photo to determine where it was taken.

Step 1: Identify visual cues (architecture, vegetation, road layout, terrain, traffic patterns) and estimate a rough location based ONLY on these visual cues.
Step 2: Read any text in the image and note what location it suggests.
Step 3: If the visual cues and text suggest DIFFERENT locations, explain the conflict and decide which source is more reliable.
Step 4: Provide your final GPS coordinates, prioritizing the most reliable evidence.

You MUST provide your best estimate even if uncertain.
Output in this exact format (one item per line):
Visual: [region/country from visual cues]
Text: [location from text, or "none" if no text, or "consistent" if same as visual]
Conflict: [Yes/No]
Trusted: [Visual/Textual/Both]
Final: (Latitude, Longitude)"""

# =============================================================================
#  Attack generation (scene-text replacement)
#  Shared by benchmark construction and the supplementary prompt card.
# =============================================================================


def build_attack_generation_prompt(location_info: dict | None = None) -> str:
    """Build the exact prompt used to generate benchmark interventions."""

    location_context = ""
    if location_info:
        parts = []
        for key in ("city", "county", "province", "country"):
            value = str(location_info.get(key, "")).strip()
            if value:
                parts.append(value)
        if parts:
            location_context = f"\nGround-truth location: {', '.join(parts)}\n"

    return f"""Analyze legible text in this street-view image for a geo-localization robustness study.
{location_context}
Task:
1. Identify the top 1--3 geo-informative texts in the image.
   Priority: street/road signs > city/district names > store names with location cues > other text.
   Return 1--3 entries. Skip illegible, tiny, or irrelevant text such as license plates and watermarks.
   If no legible text is found, return {{"texts": []}}.

2. For each selected text, describe where it appears in the image.

3. For each selected text, generate three replacements:
   - similar: plausible in the same region/language and should not change the perceived location;
   - random: unrelated text in a different language/script, using diverse choices;
   - adversarial: a real place name from a different continent, using diverse targets.

Rules:
- Keep replacements short, approximately matching the original length (about 15 characters maximum).
- Use unique replacements for separate text entries.
- Match the original script/style where possible, except that random deliberately changes script.

Return JSON only:
{{"texts": [{{"original_text": "...", "text_location": "...", "attacks": {{"similar": "...", "random": "...", "adversarial": "..."}}}}]}}"""


# The paper card uses placeholders for the dynamic source-location metadata.
ATTACK_GENERATION_PROMPT = build_attack_generation_prompt(
    {"city": "<city>", "province": "<province>", "country": "<country>"}
)

# =============================================================================
#  Cross-task generalization
# =============================================================================

GENERALIZATION_TASKS = {
    # Task A: Scene-Text Consistency Judgment
    "consistency": {
        "prompt": """You are given an image containing both scene context and possibly visible text.

Task:
1. Describe the main visual cues in the scene (architecture, vegetation, terrain, road layout, traffic conventions).
2. Describe what the visible text suggests, if any (signs, storefronts, addresses, license plates).
3. Decide whether the text is consistent with the visual scene.

Output ONLY valid JSON:
{
  "visual_cues": "brief description of visual evidence",
  "text_cues": "brief description of text and what location it suggests, or 'none' if no readable text",
  "consistency": "Consistent or Conflict",
  "reason": "brief explanation of why consistent or conflicting"
}""",
        "description": "Scene-Text Consistency Judgment",
    },
    # Task B: Coarse Region Reasoning under Conflict
    "country": {
        "prompt": """You are given an image.

Task:
1. Infer the most likely country or broad region based on the visual scene only (architecture, vegetation, terrain, road layout, traffic conventions, vehicle types).
2. Note any visible text (signs, storefronts, license plates) and what location it suggests.
3. If the scene and text disagree, explain which source is more reliable and why.
4. Give your final answer for the country or region.

Output ONLY valid JSON:
{
  "scene_region": "country/region inferred from visual cues only",
  "text_region": "country/region suggested by text, or 'none' if no text",
  "trusted_source": "Visual or Textual or Both",
  "final_region": "your final country/region answer",
  "reason": "brief explanation"
}""",
        "description": "Coarse Region Reasoning under Conflict",
    },
    "language": {
        "prompt": """Examine this photo carefully. What language(s) can you read in any visible text (signs, labels, etc.)?

Output ONLY valid JSON:
{"languages": ["language1", "language2"], "text_found": true, "examples": ["example text"], "reasoning": "brief explanation"}""",
        "description": "Language identification",
    },
    "cardinal": {
        "prompt": """Based on visual cues in this photo (sun position, shadows, vegetation patterns, compass indicators), estimate which cardinal direction the camera is facing.

Output ONLY valid JSON:
{"direction": "N/S/E/W/NE/NW/SE/SW", "confidence": "high/medium/low", "cues": ["cue1", "cue2"]}""",
        "description": "Cardinal direction estimation",
    },
}
