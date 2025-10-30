from __future__ import annotations

import os
from typing import List, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_with_vader(texts: List[str]) -> Tuple[float, str]:
    """Return sentiment score in [0,1] and label."""
    if not texts:
        return 0.5, "neutral"
    an = SentimentIntensityAnalyzer()
    scores = [an.polarity_scores(t)["compound"] for t in texts]
    mean = sum(scores) / len(scores)
    label = "positive" if mean > 0.05 else "negative" if mean < -0.05 else "neutral"
    # scale [-1,1] -> [0,1]
    conf = (mean + 1) / 2
    return conf, label


def summarize_openai(texts: List[str], model: str = "gpt-4o-mini") -> Optional[dict]:
    """
    Summarize a set of headlines/paragraphs using OpenAI if key is present.
    Returns dict with sentiment_score [0,1], label, and summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a market news assistant. Given recent headlines, "
        "produce a concise 4-bullet summary and an overall sentiment label "
        "(positive/neutral/negative) with a confidence in [0,1]. Return JSON with\n"
        "{summary, label, confidence}."
    )
    joined = "\n".join(f"- {t}" for t in texts[:20])
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Headlines:\n{joined}"},
    ]

    try:
        resp = client.chat.completions.create(model=model, messages=msg, temperature=0.2)
        content = resp.choices[0].message.content
    except Exception:
        return None

    # Try to parse JSON; fallback to heuristic sentiment
    import json, re

    try:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else json.loads(content)
        conf = float(data.get("confidence", 0.5))
        label = str(data.get("label", "neutral")).lower()
        summary = str(data.get("summary", content))
        conf = min(max(conf, 0.0), 1.0)
        return {"sentiment_score": conf, "label": label, "summary": summary}
    except Exception:
        score, label = analyze_with_vader(texts)
        return {"sentiment_score": score, "label": label, "summary": content}

