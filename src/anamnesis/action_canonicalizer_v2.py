"""Additive v2 canonicalizer preserving the frozen v1 implementation."""

from __future__ import annotations

import re

from anamnesis.action_canonicalizer import (
    CanonicalizationChange,
    CanonicalizationResult,
)
from anamnesis.action_canonicalizer import (
    canonicalize_immediate_decision as canonicalize_v1,
)
from anamnesis.schema import Decision, ObservableEvent, ProposedAction

ACTION_CANONICALIZER_VERSION = "immediate-action-canonicalizer.v2"
_UPLOAD_DOCUMENT = re.compile(
    r"^act now:\s*upload (?:the )?(.+?) for (?:my |the )?(.+?)[.!]?$",
    re.IGNORECASE,
)
_IMPERATIVE_ARTICLE = re.compile(
    r"^(send|track|upload|photograph|submit|deliver|mail|call|book|review|check|"
    r"print|archive|scan|file|renew|pay|schedule|reserve|collect|order|notify|"
    r"contact)\s+(?:the|a|an)\s+(.+)$",
    re.IGNORECASE,
)
_DOMAIN_SUFFIX = re.compile(r"\s+(?:study|survey|project)$", re.IGNORECASE)


def canonicalize_immediate_decision_v2(
    *,
    event: ObservableEvent,
    retrospective_recall: tuple[str, ...] | None,
    decision: Decision,
) -> CanonicalizationResult:
    """Apply frozen v1, then three conservative source-grounded v2 rules."""

    base = canonicalize_v1(
        event=event,
        retrospective_recall=retrospective_recall,
        decision=decision,
    )
    if len(base.decision.actions) != 1:
        return base
    action = base.decision.actions[0]
    if action.action_key != event.id or tuple(action.evidence_event_ids) != (event.id,):
        return base
    payload = dict(action.payload)
    changes = list(base.changes)

    recipient = payload.get("recipient")
    if isinstance(recipient, str) and recipient.casefold() in event.text.casefold():
        for key in ("room", "address"):
            value = payload.get(key)
            if isinstance(value, str) and value.casefold() == recipient.casefold():
                del payload[key]
                changes.append(
                    CanonicalizationChange(kind="drop_redundant", source=key)
                )

    subject = str(payload["subject"])
    item = payload.get("item")
    project = payload.get("project")
    document_match = _UPLOAD_DOCUMENT.fullmatch(event.text.strip())
    recall_folded = "\n".join(retrospective_recall or ()).casefold()
    if (
        document_match
        and isinstance(item, str)
        and isinstance(project, str)
        and project.casefold() in recall_folded
    ):
        document = document_match.group(1).strip().casefold()
        domain = document_match.group(2).strip()
        if (
            subject.casefold() == f"upload {document}"
            and item.casefold() == domain.casefold()
        ):
            topic = _DOMAIN_SUFFIX.sub("", domain).strip().casefold()
            payload["subject"] = f"upload {topic} {document}"
            del payload["item"]
            changes.extend(
                (
                    CanonicalizationChange(
                        kind="rewrite_subject", source="subject", target="subject"
                    ),
                    CanonicalizationChange(kind="drop_redundant", source="item"),
                )
            )

    subject = str(payload["subject"])
    article_match = _IMPERATIVE_ARTICLE.fullmatch(subject)
    if article_match and subject.casefold() in event.text.casefold():
        payload["subject"] = (
            f"{article_match.group(1).casefold()} {article_match.group(2)}"
        )
        changes.append(
            CanonicalizationChange(
                kind="rewrite_subject", source="subject", target="subject"
            )
        )

    normalized = ProposedAction(
        kind=action.kind,
        action_key=action.action_key,
        payload=payload,
        summary=action.summary,
        evidence_event_ids=action.evidence_event_ids,
    )
    return CanonicalizationResult(
        decision=Decision(actions=[normalized]), changes=tuple(changes)
    )


__all__ = ["ACTION_CANONICALIZER_VERSION", "canonicalize_immediate_decision_v2"]
