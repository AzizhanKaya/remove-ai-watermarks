//! Polarity and modality cues: negation, antonym poles, constraints, capability frames, expected actions.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::facts::numbers::{NUMERIC_BOUND_RE, without_placeholders};
use crate::pyre::{Flags, group, group_span, py_regex};
use crate::unicode;

py_regex!(
    PSEUDO_NEGATION_RE,
    r"\b(?:not\s+only|not\s+just|not\s+merely|not\s+simply|not\s+solely|not\s+necessarily|without\s+difficulty)\b",
    Flags::I
);
py_regex!(
    NEGATION_TOKEN_RE,
    r"\b(?:not|no|never|none|neither|nor|without|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|mustn't|shouldn't|wouldn't|couldn't|refus(?:e|es|ed)\s+to\s+let)\b",
    Flags::I
);
py_regex!(
    UNRESTRICTED_SWING_RE,
    r"\b(?P<movement>(?:gate|door|panel) to swing (?:inward|outward)) (?P<cue>freely|without restriction)\b",
    Flags::I
);
py_regex!(CONTENT_WORD_RE, r"[^\W_]+");
py_regex!(RUSSIAN_NOT_ONLY_RE, r"\bне\s+только\b", Flags::I);

const ANTONYM_POLES: [(&[&str], &[&str]); 8] = [
    (
        &[
            "increase",
            "increases",
            "increased",
            "increasing",
            "rise",
            "rises",
            "rose",
            "grow",
            "grows",
            "grew",
        ],
        &[
            "decrease",
            "decreases",
            "decreased",
            "decreasing",
            "fall",
            "falls",
            "fell",
            "shrink",
            "shrinks",
            "shrank",
            "drop",
            "drops",
            "dropped",
        ],
    ),
    (
        &["succeed", "succeeds", "succeeded", "pass", "passes", "passed"],
        &["fail", "fails", "failed", "failing"],
    ),
    (&["always"], &["never"]),
    (
        &["before", "prior", "precede", "precedes", "preceded"],
        &["after", "subsequent", "follow", "follows", "followed"],
    ),
    (
        &["above", "over", "exceed", "exceeds", "exceeded"],
        &["below", "under"],
    ),
    (
        &["true", "correct", "accurate", "valid"],
        &["false", "incorrect", "inaccurate", "invalid"],
    ),
    (
        &["allowed", "permitted", "legal", "authorized"],
        &["forbidden", "prohibited", "illegal", "banned", "unauthorized"],
    ),
    (&["open", "opened"], &["closed", "shut"]),
];

const CLAIM_STOPWORDS: [&str; 30] = [
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "had", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "which", "with",
];

pub fn has_active_negation(text: &str) -> bool {
    let text = text.replace('\u{2019}', "'");
    let scrubbed = without_placeholders(&text);
    let scrubbed = PSEUDO_NEGATION_RE.sub(" ", &scrubbed);
    // Quantified "no fewer/more than" is a bound, not predicate negation.
    let scrubbed = NUMERIC_BOUND_RE.sub(" ", &scrubbed);
    NEGATION_TOKEN_RE.is_match(&scrubbed)
}

/// Compare polarity after matching the scope of an unrestricted physical swing.
pub fn preserves_rewrite_negation(source: &str, candidate: &str) -> bool {
    let original: Vec<_> = UNRESTRICTED_SWING_RE.captures_iter(source).collect();
    let revised: Vec<_> = UNRESTRICTED_SWING_RE.captures_iter(candidate).collect();
    let movement = |caps: &fancy_regex::Captures<'_, str>| group(caps, "movement").map(unicode::casefold);
    if original.len() == 1 && revised.len() == 1 && movement(&original[0]) == movement(&revised[0]) {
        let keep_movement =
            |caps: &fancy_regex::Captures<'_, str>| group(caps, "movement").unwrap_or("").to_owned();
        let source = UNRESTRICTED_SWING_RE.sub_with(source, keep_movement);
        let candidate = UNRESTRICTED_SWING_RE.sub_with(candidate, keep_movement);
        return has_active_negation(&source) == has_active_negation(&candidate);
    }
    has_active_negation(source) == has_active_negation(candidate)
}

/// Lowercased content words of a sentence, minus stopwords and one-letter tokens.
pub fn sentence_content_words(sentence: &str) -> BTreeSet<String> {
    let lowered = unicode::lower(&without_placeholders(sentence));
    CONTENT_WORD_RE
        .findall(&lowered)
        .into_iter()
        .filter(|word| word.chars().count() >= 2 && !CLAIM_STOPWORDS.contains(word))
        .map(str::to_owned)
        .collect()
}

pub(crate) fn has_antonym_pole_inversion(source: &BTreeSet<String>, candidate: &BTreeSet<String>) -> bool {
    let any = |words: &BTreeSet<String>, pole: &[&str]| pole.iter().any(|word| words.contains(*word));
    ANTONYM_POLES.iter().any(|(pole_a, pole_b)| {
        let (src_a, src_b) = (any(source, pole_a), any(source, pole_b));
        let (cand_a, cand_b) = (any(candidate, pole_a), any(candidate, pole_b));
        (src_a && !src_b && cand_b && !cand_a) || (src_b && !src_a && cand_a && !cand_b)
    })
}

py_regex!(
    APPROVAL_RE,
    r"\b(?:approv(?:e[ds]?|ing|al|als)|authori[sz](?:e[ds]?|ing|ation|ations)|одобрен(?:а|о|ы)?|одобрени[ея]|одобр(?:ить|или|ил[аи]?|яет|яют)|утвержд[её]н(?:а|о|ы)?|утверд(?:ить|или|ил[аи]?|ят|ит))\b",
    Flags::I
);
py_regex!(
    PERMISSION_RE,
    r"\b(?:allow(?:s|ed|ing)?|permit(?:s|ted|ting)?|permissions?|разреш(?:ать|ить|ают|ает|ил[аи]?|или|[её]н(?:а|о|ы)?)|разрешени[ея])\b",
    Flags::I
);
py_regex!(
    RESPONSIBILITY_RE,
    r"\b(?:responsib(?:le|ility|ilities)|accountab(?:le|ility)|in\s+charge\s+of|ответственност(?:ь|и|ью)|отвеча(?:ет|ют|ешь|ем|ете|л[аи]?|ли)\s+за)\b",
    Flags::I
);
py_regex!(
    RESTRICTION_RE,
    r"\b(?:only|merely|solely|exclusively|restricted\s+to|limited\s+to|только|лишь|исключительно)\b",
    Flags::I
);
py_regex!(
    REQUIREMENT_RE,
    r"\b(?:must|mustn\x27t|require(?:s|d)?|necessary|necessity\s+for|indispensable|mandatory|obligatory|need(?:s|ed|ing)?|necessitat(?:e[ds]?|ing)|essential\s+(?:tools?|instruments?|to\s+(?:know|grasp|understand))|обязательна|обязательно|обязательны|обязательный|обязательная|обязательное|обязательные|требуется|необходимо|должен|должна|должно|должны)\b",
    Flags::I
);
py_regex!(
    IMPOSSIBLE_WITHOUT_RE,
    r"\b(?:без\b[^.!?;]{0,120}\b(?:невозможен|невозможна|невозможно|невозможны)|(?:невозможен|невозможна|невозможно|невозможны)\b[^.!?;]{0,120}\bбез)\b",
    Flags::I
);

/// A selected explicit constraint family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Constraint {
    Approval,
    Permission,
    Requirement,
    Responsibility,
    Restriction,
}

/// Selected explicit constraints, without proving their scope or role.
pub fn extract_rewrite_constraints(text: &str) -> BTreeSet<Constraint> {
    let content = without_placeholders(&text.replace('\u{2019}', "'"));
    let content = PSEUDO_NEGATION_RE.sub(" ", &content);
    let content = RUSSIAN_NOT_ONLY_RE.sub(" ", &content);
    let patterns = [
        (Constraint::Approval, &*APPROVAL_RE),
        (Constraint::Permission, &*PERMISSION_RE),
        (Constraint::Responsibility, &*RESPONSIBILITY_RE),
        (Constraint::Restriction, &*RESTRICTION_RE),
        (Constraint::Requirement, &*REQUIREMENT_RE),
        (Constraint::Requirement, &*IMPOSSIBLE_WITHOUT_RE),
    ];
    patterns
        .into_iter()
        .filter(|(_, pattern)| pattern.is_match(&content))
        .map(|(name, _)| name)
        .collect()
}

py_regex!(
    EXPECTED_OR_REQUIRED_ACTION_RE,
    r"(?:\A\s*|(?<=[.!?;])\s+)(?P<actor>the[ \t]+[a-z'-]+(?:[ \t]+[a-z'-]+){0,4})[ \t]+(?:is|are|was|were|will[ \t]+be)[ \t]+(?P<cue>expected|required)[ \t]+to[ \t]+(?P<action>[a-z][a-z'-]*(?:[ \t]+(?!(?:before|prior|after|until|when|once)\b)[a-z][a-z'-]*){0,11})(?=[ \t]+(?:before|prior[ \t]+to|after|until|when|once)\b|[,.!?;:\r\n]|\Z)",
    Flags::I
);

/// Affirmative expected/required clauses whose cue changed for a matching actor and action.
pub fn changes_explicit_expected_action(source: &str, candidate: &str) -> bool {
    let extract = |value: &str| {
        let mut cues: HashMap<(String, String), HashSet<String>> = HashMap::new();
        for caps in EXPECTED_OR_REQUIRED_ACTION_RE.captures_iter(value) {
            let actor = unicode::fold_words(group(&caps, "actor").unwrap_or(""));
            let action = unicode::fold_words(group(&caps, "action").unwrap_or(""))
                .replace(" agreed-upon ", " agreed upon ");
            cues.entry((actor, action))
                .or_default()
                .insert(unicode::casefold(group(&caps, "cue").unwrap_or("")));
        }
        cues
    };
    let (before, after) = (extract(source), extract(candidate));
    before.iter().any(|(key, left)| {
        after
            .get(key)
            .is_some_and(|right| left.len() == 1 && right.len() == 1 && left != right)
    })
}

py_regex!(
    TECHNICAL_CAPABILITY_RE,
    r"(?:^\s*|(?<=[.!?;])\s+)(?P<subject>(?:(?:the|an?)\s+)?(?:(?:new|old|updated|digital|shared|automated|online|electronic)\s+)*(?:system|software|application|feature|device|equipment|calculator))\s+(?P<cue>allow(?:s|ed)?|enabl(?:e|es|ed))\s+(?P<actor>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,4})\s+to\s+(?P<action>record|document|log|store|save|calculate|compute|measure|display|show)\s+(?P<object>[^,.!?;:\n]{1,120})(?=[,.!?;:\n]|$)",
    Flags::I
);
py_regex!(
    FILM_CAPTURE_CAPABILITY_RE,
    r"(?:^\s*|(?<=[.!?;])\s+)this (?:continuous|ongoing) mechanical advanc(?:e|ement) (?:is what )?(?P<cue>allows|enables) (?:a|the) film camera to (?:capture|record) the (?:same|identical) scene (?:multiple times|repeatedly) without (?:requiring )?manual rewinding(?=[,.!?;:]|$)",
    Flags::I
);
py_regex!(
    MOISTURE_EVAPORATION_CAPABILITY_RE,
    r"\b(?:conduit|channel) (?P<cue>for|that (?:allows|enables)) moisture to evaporate\b",
    Flags::I
);
py_regex!(
    AIR_CIRCULATION_CAPABILITY_RE,
    r",\s+(?P<cue>allowing|enabling) air (?:circulation|movement)(?: (?P<location>throughout the [a-z-]+))?(?=[,.!?;:]|\s+to\b|$)",
    Flags::I
);
py_regex!(
    MECHANICAL_OPEN_CLOSE_RE,
    r"\bmechanical (?:device|apparatus|mechanism) that (?P<cue>allows for|facilitates|enables) the (?:effortless|smooth) opening and closing of (?P<object>[^,.!?;:\n]{1,80})(?=[,.!?;:]|$)",
    Flags::I
);
py_regex!(
    FIBER_SLIDING_RE,
    r"\b(?:fibers|fibres),\s+(?P<cue>causing|allowing|enabling) them to slide (?:past|over) (?:one another|each other) (?:more easily|with greater ease)(?=[,.!?;:]|$)",
    Flags::I
);
py_regex!(
    SUBJECT_SEARCH_FILTER_RE,
    r"(?:^\s*|(?<=[.!?;])\s+)subjects (?:provide|offer) more (?:specific|precise) (?:filters|filtering options),\s+(?P<cue>allowing|enabling) users to (?:narrow|refine) their searches(?: to a (?:narrower scope|specific area))?(?=[,.!?;:]|$)",
    Flags::I
);

fn capability_action(action: &str) -> &'static str {
    match action {
        "record" | "document" | "log" => "record",
        "store" | "save" => "store",
        "calculate" | "compute" => "calculate",
        "measure" => "measure",
        _ => "display",
    }
}

type FrameKey = (String, String, String, String);
type Frames = HashMap<FrameKey, Vec<(usize, usize, bool)>>;

fn capability_frames(text: &str) -> Frames {
    let mut found: Frames = HashMap::new();
    let mut push = |key: FrameKey, caps: &fancy_regex::Captures<'_, str>, allow: bool| {
        if let Some(span) = group_span(caps, "cue") {
            found.entry(key).or_default().push((span.start, span.end, allow));
        }
    };
    let cue = |caps: &fancy_regex::Captures<'_, str>| unicode::casefold(group(caps, "cue").unwrap_or(""));
    for caps in TECHNICAL_CAPABILITY_RE.captures_iter(text) {
        let [subject, actor, action, object] = ["subject", "actor", "action", "object"]
            .map(|name| unicode::fold_words(group(&caps, name).unwrap_or("")));
        let object: Vec<&str> = object
            .split(' ')
            .map(|word| match word {
                "extra" => "additional",
                "details" => "information",
                other => other,
            })
            .collect();
        let key = (
            subject,
            actor,
            capability_action(&action).to_owned(),
            object.join(" "),
        );
        push(key, &caps, cue(&caps).starts_with("allow"));
    }
    let fixed = |a: &str, b: &str, c: &str, d: &str| (a.to_owned(), b.to_owned(), c.to_owned(), d.to_owned());
    for caps in FILM_CAPTURE_CAPABILITY_RE.captures_iter(text) {
        push(
            fixed(
                "mechanical film advance",
                "film camera",
                "capture",
                "repeat scene without manual rewind",
            ),
            &caps,
            cue(&caps) == "allows",
        );
    }
    for caps in MOISTURE_EVAPORATION_CAPABILITY_RE.captures_iter(text) {
        push(
            fixed("conduit", "moisture", "evaporate", ""),
            &caps,
            cue(&caps) == "that allows",
        );
    }
    for caps in AIR_CIRCULATION_CAPABILITY_RE.captures_iter(text) {
        let location = unicode::casefold(group(&caps, "location").unwrap_or(""));
        push(
            fixed("", "air", "circulate", &location),
            &caps,
            cue(&caps) == "allowing",
        );
    }
    for caps in MECHANICAL_OPEN_CLOSE_RE.captures_iter(text) {
        let object = unicode::fold_words(group(&caps, "object").unwrap_or(""));
        push(
            fixed("mechanical device", "", "open and close smoothly", &object),
            &caps,
            cue(&caps) == "allows for",
        );
    }
    for caps in FIBER_SLIDING_RE.captures_iter(text) {
        push(
            fixed("", "fibers", "slide past one another more easily", ""),
            &caps,
            cue(&caps) == "allowing",
        );
    }
    for caps in SUBJECT_SEARCH_FILTER_RE.captures_iter(text) {
        push(
            fixed("subject filters", "users", "narrow", "their searches"),
            &caps,
            cue(&caps) == "allowing",
        );
    }
    found
}

/// Erase matched singleton allow/enable cue pairs in temporary comparison strings.
pub fn constraint_texts_for_capability_frames(source: &str, candidate: &str) -> (String, String) {
    let (source_frames, candidate_frames) = (capability_frames(source), capability_frames(candidate));
    let mut source_spans = Vec::new();
    let mut candidate_spans = Vec::new();
    for (key, left) in &source_frames {
        if let Some(right) = candidate_frames.get(key)
            && left.len() == 1
            && right.len() == 1
            && left[0].2 != right[0].2
        {
            source_spans.push((left[0].0, left[0].1));
            candidate_spans.push((right[0].0, right[0].1));
        }
    }
    let erase = |text: &str, mut spans: Vec<(usize, usize)>| {
        spans.sort_unstable_by(|a, b| b.cmp(a));
        let mut text = text.to_owned();
        for (start, end) in spans {
            text.replace_range(start..end, " ");
        }
        text
    };
    (erase(source, source_spans), erase(candidate, candidate_spans))
}
