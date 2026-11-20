"""Qualified presentation labels; these do not calibrate the underlying models."""

import math


def _check(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score")


def generation_conclusion(score):
    _check(score)
    if score >= .9:
        return "Likely AI-generated or AI-edited."
    if score <= .1:
        return "Likely not fully AI-generated. AI-assisted edits may still be present."
    return "Unable to determine whether AI was used."
