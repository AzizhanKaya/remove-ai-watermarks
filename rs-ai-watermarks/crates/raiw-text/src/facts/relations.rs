//! Selected English/Russian relation frames: seasons, records, signals, memberships, events, qualifiers.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use crate::facts::names::cached_regex;
use crate::facts::numbers::without_placeholders;
use crate::pyre::{Flags, PyRegex, escape, group, py_regex, whole};
use crate::unicode;

// ── Seasons ──

py_regex!(
    AUTUMN_RE,
    r"\b(?:autumns?|autumntime|осень|осенью|осени)\b",
    Flags::I
);
py_regex!(
    WINTER_RE,
    r"\b(?:winters?|wintertime|зима|зимой|зиму|зимы|зиме)\b",
    Flags::I
);
py_regex!(SUMMER_RE, r"\b(?:summers?|summertime|лето|летом)\b", Flags::I);
py_regex!(
    SPRING_RE,
    r"\b(?:springtime|весна|весной|весну|весны|весне)\b",
    Flags::I
);
py_regex!(FALL_ALIAS_RE, r"\bfall\b", Flags::I);
py_regex!(SPRING_ALIAS_RE, r"\bspring\b", Flags::I);
py_regex!(
    SPRING_TO_AUTUMN_RE,
    r"\bfrom\s+(?:the\s+)?spring\s+(?:to|through|until)\s+(?:the\s+)?(?:fall|autumn)\b",
    Flags::I
);

pub fn explicit_seasons(text: &str) -> BTreeSet<&'static str> {
    let content = without_placeholders(text);
    let mut seasons: BTreeSet<&'static str> = [
        ("autumn", &*AUTUMN_RE),
        ("winter", &*WINTER_RE),
        ("summer", &*SUMMER_RE),
        ("spring", &*SPRING_RE),
    ]
    .into_iter()
    .filter(|(_, pattern)| pattern.is_match(&content))
    .map(|(name, _)| name)
    .collect();
    if SPRING_TO_AUTUMN_RE.is_match(&content) {
        seasons.extend(["spring", "autumn"]);
    }
    seasons
}

/// Selected season mentions, without inferring ambiguous spring/fall senses.
pub fn preserves_explicit_seasons(source: &str, candidate: &str) -> bool {
    let (source_seasons, candidate_seasons) = (explicit_seasons(source), explicit_seasons(candidate));
    // A bare alias can explain an explicit counterpart, but is not independently evidence.
    for (differences, other) in [
        (source_seasons.difference(&candidate_seasons), candidate),
        (candidate_seasons.difference(&source_seasons), source),
    ] {
        for season in differences {
            let alias = match *season {
                "autumn" => &*FALL_ALIAS_RE,
                "spring" => &*SPRING_ALIAS_RE,
                _ => return false,
            };
            if !alias.is_match(&without_placeholders(other)) {
                return false;
            }
        }
    }
    true
}

// ── Record distribution ──

const RECORD: &str = r"(?:notebooks?|ledgers?|logbooks?|registers?|logs?)\b(?![-‐‑]\w)(?!\s+(?:entr(?:y|ies)|pages?|rows?|sections?|numbers?|codes?|covers?|labels?|titles?)\b)";
const RECORD_MODIFIERS: &str =
    r"(?:(?:own|separate|individual|personal|personalized|dedicated|simple|new)\s+)*";

static SHARED_RECORD_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"\b(?:shared|communal|common|collective)\s+(?:(?:simple|central|main|single|written|electronic|digital|paper)\s+)*{RECORD}"
        ),
        Flags::I,
    )
});
static PER_ITEM_RECORD_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"\b(?:each|every)\s+(?:[a-z][a-z'-]*\s+){{0,3}}?(?:has|have|keeps?|gets?|receives?)\s+(?:its|their|his|her|a|an)\s+{RECORD_MODIFIERS}{RECORD}|\beach\s+(?:(?:is|was)\s+)?accompanied\s+by\s+(?:its|their|a|an)\s+{RECORD_MODIFIERS}{RECORD}|\b(?:separate|individual|dedicated)\s+{RECORD}\s+(?:accompanies?\s+(?:each|every)|per\s+[a-z])"
        ),
        Flags::I,
    )
});

/// Selected explicit English shared/per-item record containers.
pub fn extract_explicit_record_distribution(text: &str) -> BTreeSet<&'static str> {
    let prose = without_placeholders(text);
    let mut modes = BTreeSet::new();
    if SHARED_RECORD_RE.is_match(&prose) {
        modes.insert("shared");
    }
    if PER_ITEM_RECORD_RE.is_match(&prose) {
        modes.insert("per_item");
    }
    modes
}

// ── Signal directions and membership stances ──

static SIGNAL_DIRECTION_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    let participants =
        r"(?:community|volunteers?|owners?|council|committee|board|residents?|neighbors?|team|group)";
    let modifiers = r"(?:(?:original|local)\s+)?";
    let boundary = r"(?=$|\s*[,.;:!?]|\s+(?:and|or|but|before|after|while|when|who|which|that|with|without|in|for|on|as|is|are|was|were|has|have|had|gave|gives|give|received|receives|receive)\b)";
    PyRegex::new(
        &format!(
            r"\b(nods?|approval|permission|consent)(?:\s+in\s+return)?\s+(from|to)\s+(?:the\s+)?({modifiers}{participants})\b(?![-‐‑]\w){boundary}"
        ),
        Flags::I,
    )
});

/// `(signal, participant)` keyed explicit `from`/`to` directions.
pub type SignalDirections = BTreeMap<(String, String), BTreeSet<String>>;

pub fn extract_explicit_signal_directions(text: &str) -> SignalDirections {
    let mut directions = SignalDirections::new();
    let prose = without_placeholders(text);
    for caps in SIGNAL_DIRECTION_RE.captures_iter(&prose) {
        let field = |index: usize| caps.get(index).map_or("", |m| m.as_str());
        let signal = unicode::casefold(field(1));
        let signal = signal.strip_suffix('s').unwrap_or(&signal).to_owned();
        directions
            .entry((signal, unicode::fold_words(field(3))))
            .or_default()
            .insert(unicode::casefold(field(2)));
    }
    directions
}

py_regex!(
    MEMBERSHIP_RE,
    r"\b(?P<subject>(?:the|an?|this|that)[ \t]+[a-z][a-z'-]*(?:[ \t]+[a-z][a-z'-]*){0,3}?)\s+(?:is|are|was|were)\s+(?P<relation>part\s+of|(?:a\s+)?member\s+of|critical\s+of)\s+(?P<object>[^,.:;!?\r\n]{1,120})(?=[,.:;!?\r\n]|$)",
    Flags::I
);
py_regex!(PARTICIPANT_DETERMINER_RE, r"^(?:the|an?|this|that)\s+");

pub type MembershipStances = BTreeMap<(String, String), BTreeSet<&'static str>>;

pub fn extract_explicit_membership_stances(text: &str) -> MembershipStances {
    let participant = |value: &str| {
        PARTICIPANT_DETERMINER_RE
            .sub("", &unicode::fold_words(value))
            .into_owned()
    };
    let mut frames = MembershipStances::new();
    for caps in MEMBERSHIP_RE.captures_iter(&without_placeholders(text)) {
        let key = (
            participant(group(&caps, "subject").unwrap_or("")),
            participant(group(&caps, "object").unwrap_or("")),
        );
        let relation = if unicode::casefold(group(&caps, "relation").unwrap_or("")).starts_with("critical") {
            "criticism"
        } else {
            "membership"
        };
        frames.entry(key).or_default().insert(relation);
    }
    frames
}

// ── Type coverage and handovers ──

pub(crate) const TYPE_COVERAGE_PATTERN: &str = r"(?<![\w\-‐‑])(?:(?:each|every)\s+(?:[a-z][a-z'-]*\s+){0,2}type|all\s+(?:[a-z][a-z'-]*\s+){0,2}types)(?![\w\-‐‑])";
py_regex!(TYPE_COVERAGE_RE, TYPE_COVERAGE_PATTERN, Flags::I);

pub fn explicit_type_coverage_count(text: &str) -> usize {
    TYPE_COVERAGE_RE.count(&without_placeholders(text))
}

py_regex!(
    HANDOVER_RE,
    r"\b(?P<action>offered|handed)\s+(?P<recipient>her|him|them|me|us|you)\s+(?:an?|the)\s+(?:(?:folded|sealed|open|closed|empty|full|wrapped|small|large|paper|spare)\s+){0,2}(?P<item>book|cup|map|key|packet|letter|envelope|ticket|document|bag|folder|scarf|parcel|tool|box|coin|bottle)\b",
    Flags::I
);

pub type Handovers = BTreeMap<(String, String), BTreeSet<String>>;

pub fn extract_handover_status(text: &str) -> Handovers {
    let mut events = Handovers::new();
    let folded = unicode::casefold(text);
    for caps in HANDOVER_RE.captures_iter(&folded) {
        let field = |name: &str| group(&caps, name).unwrap_or("").to_owned();
        events
            .entry((field("recipient"), field("item")))
            .or_default()
            .insert(field("action"));
    }
    events
}

// ── Actor apposition ──

const SENTENCE_START: &str = r"(?:\A|(?<=[.!?\n]))[ \t]*";
const APPOSITIVE_ACTIONS: [(&str, &str); 7] = [
    ("greeted", "greeted|welcomed"),
    ("welcomed", "greeted|welcomed"),
    ("handed", "handed|given|gave"),
    ("gave", "handed|given|gave"),
    ("escorted", "escorted"),
    ("interviewed", "interviewed"),
    ("thanked", "thanked"),
];
static ACTOR_APPOSITION_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    let actions: Vec<&str> = APPOSITIVE_ACTIONS.iter().map(|(action, _)| *action).collect();
    PyRegex::new(
        &format!(
            r"{SENTENCE_START}(?P<title>the(?: [a-z][a-z'-]*){{1,6}}), (?P<description>an? [a-z][a-z'-]* with [^,.;:!?\n]{{1,120}}), (?P<action>{})\b",
            actions.join("|")
        ),
        Flags::NONE,
    )
});
py_regex!(HORIZONTAL_SPACE_RE, r"[ \t]+");
py_regex!(INDEFINITE_PROPERTY_RE, r"an? ");

fn compact_folded(text: &str) -> String {
    unicode::casefold(&HORIZONTAL_SPACE_RE.sub(" ", &without_placeholders(text)))
}

/// Contrasts between an actor's apposition and a separate participant.
pub fn changes_actor_apposition(source: &str, candidate: &str) -> bool {
    let (source, candidate) = (compact_folded(source), compact_folded(candidate));
    for caps in ACTOR_APPOSITION_RE.captures_iter(&source) {
        let description = group(&caps, "description").unwrap_or("");
        let action_word = group(&caps, "action").unwrap_or("");
        let title = escape(group(&caps, "title").unwrap_or(""));
        let mut descriptions = vec![escape(description)];
        let (head, details) = description.split_once(" with ").unwrap_or((description, ""));
        let properties: Vec<&str> = details.split(" and ").collect();
        if properties.len() == 2
            && properties
                .iter()
                .all(|prop| INDEFINITE_PROPERTY_RE.match_at(prop, 0).is_some())
        {
            descriptions.push(escape(&format!(
                "{head} with {} and {}",
                properties[1], properties[0]
            )));
        }
        let descriptor = descriptions.join("|");
        let action = APPOSITIVE_ACTIONS
            .iter()
            .find(|(word, _)| *word == action_word)
            .map_or("", |(_, forms)| *forms);
        let mut passive_argument = r" to (?:her|him|them|me|us|you|the(?: [a-z][a-z'-]*){1,5})".to_owned();
        if matches!(action_word, "handed" | "gave") {
            passive_argument.push_str(r"| (?:an?|the) [^,.;:!?\n]{1,120}?");
        }
        let independent = format!(
            r"{SENTENCE_START}(?:{descriptor}) (?:(?:{action}) |(?:was|were) (?:{action})(?:{passive_argument})? by ){title}(?=$|[,.!?;:\n]| (?:at|in|on)\b)"
        );
        let pattern = cached_regex(&independent, Flags::NONE);
        if pattern.is_match(&candidate) && !pattern.is_match(&source) {
            return true;
        }
    }
    false
}

// ── Event phases ──

const EVENT_VERB_FORMS: [(&str, &str, &str); 6] = [
    ("deliver", "delivered", "delivering"),
    ("repair", "repaired", "repairing"),
    ("reopen", "reopened", "reopening"),
    ("restore", "restored", "restoring"),
    ("submit", "submitted", "submitting"),
    ("unlock", "unlocked", "unlocking"),
];
const EVENT_NOUN: &str = r"(?:the|an?) [a-z][a-z'-]*(?: [a-z][a-z'-]*){0,6}?";
const EVENT_END: &str = r"(?= *(?:[.!;,\r\n]|$))";

fn event_start() -> String {
    format!(r"{SENTENCE_START}(?:(?:as|after|when) [^,.!?;:\r\n]{{1,120}}, )?")
}

fn event_actor() -> String {
    format!("(?:i|we|you|he|she|they|{EVENT_NOUN})")
}

fn verb_list(select: fn(&(&'static str, &'static str, &'static str)) -> &'static str) -> String {
    EVENT_VERB_FORMS.iter().map(select).collect::<Vec<_>>().join("|")
}

static ACTIVE_EVENT_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            "{}(?P<actor>{}) (?:(?P<stage>planned to|intended to|tried to|attempted to|managed to) (?P<base>{})|succeeded in (?P<gerund>{})|(?P<past>{})) (?P<object>{EVENT_NOUN}){EVENT_END}",
            event_start(),
            event_actor(),
            verb_list(|forms| forms.0),
            verb_list(|forms| forms.2),
            verb_list(|forms| forms.1)
        ),
        Flags::NONE,
    )
});
static PASSIVE_EVENT_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            "{}(?P<object>{EVENT_NOUN}) (?:was|were) (?P<past>{}) by (?P<actor>{EVENT_NOUN}){EVENT_END}",
            event_start(),
            verb_list(|forms| forms.1)
        ),
        Flags::NONE,
    )
});
static SHELTER_EVENT_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"{}(?P<actor>{}) (?P<action>found|took|sought|looked for|searched for) (?:shelter|refuge)(?P<location> (?:in|under|at|near|beneath|inside|on) [^,.!?;:\r\n]{{1,120}})?{EVENT_END}",
            event_start(),
            event_actor()
        ),
        Flags::NONE,
    )
});

fn event_base(verb: &str) -> &'static str {
    EVENT_VERB_FORMS
        .iter()
        .find(|(base, past, gerund)| verb == *base || verb == *past || verb == *gerund)
        .map_or("", |(base, _, _)| *base)
}

pub type EventPhases = BTreeMap<(String, String, String), BTreeSet<&'static str>>;

/// Selected explicitly bound English actions and their stated phases.
pub fn extract_event_phases(text: &str) -> EventPhases {
    let prose = compact_folded(text);
    let mut phases = EventPhases::new();
    for caps in ACTIVE_EVENT_RE.captures_iter(&prose) {
        let verb = group(&caps, "base")
            .or_else(|| group(&caps, "gerund"))
            .or_else(|| group(&caps, "past"))
            .unwrap_or("");
        let key = (
            group(&caps, "actor").unwrap_or("").to_owned(),
            event_base(verb).to_owned(),
            group(&caps, "object").unwrap_or("").to_owned(),
        );
        let stage = match group(&caps, "stage") {
            Some("planned to" | "intended to") => "plan",
            Some("tried to" | "attempted to") => "attempt",
            _ => "complete",
        };
        phases.entry(key).or_default().insert(stage);
    }
    for caps in PASSIVE_EVENT_RE.captures_iter(&prose) {
        let key = (
            group(&caps, "actor").unwrap_or("").to_owned(),
            event_base(group(&caps, "past").unwrap_or("")).to_owned(),
            group(&caps, "object").unwrap_or("").to_owned(),
        );
        phases.entry(key).or_default().insert("complete");
    }
    for caps in SHELTER_EVENT_RE.captures_iter(&prose) {
        let location = unicode::strip(group(&caps, "location").unwrap_or("")).to_owned();
        let key = (
            group(&caps, "actor").unwrap_or("").to_owned(),
            "shelter".to_owned(),
            location,
        );
        let phase = if matches!(group(&caps, "action"), Some("found" | "took")) {
            "complete"
        } else {
            "attempt"
        };
        phases.entry(key).or_default().insert(phase);
    }
    phases
}

// ── Influence qualification ──

static INFLUENCE_CLAUSE_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"(?:{SENTENCE_START}(?:(?P<condition>in [a-z][a-z '-]{{0,60}}), )?|(?:understand|recognize) that )(?P<factor>{EVENT_NOUN}) (?:(?P<modal>may|might|could|can) )?(?:(?P<adverb>significantly|substantially|materially|strongly|directly|always|necessarily) )?(?P<verb>(?:affect|impact|influence)s?) (?P<effect>how [a-z][^,.!?;:\r\n]{{1,120}}?)(?= (?:and|or) (?:ultimately )?how\b|[,.!?;:\r\n]|$)"
        ),
        Flags::NONE,
    )
});
const SURFACE_ABSORPTION_LEVEL: &str = "the level of surface absorption in ";

pub type Qualifications = BTreeMap<(String, String, String), BTreeSet<&'static str>>;

/// Selected influence claims with a literal factor, effect and condition.
pub fn extract_influence_qualifications(text: &str) -> Qualifications {
    let prose = compact_folded(text);
    let mut found = Qualifications::new();
    for caps in INFLUENCE_CLAUSE_RE.captures_iter(&prose) {
        let modal = group(&caps, "modal");
        if modal.is_some() && group(&caps, "verb").is_some_and(|verb| verb.ends_with('s')) {
            continue;
        }
        let mut factor = group(&caps, "factor").unwrap_or("").to_owned();
        if let Some(rest) = factor.strip_prefix(SURFACE_ABSORPTION_LEVEL) {
            factor = format!("the surface absorbency of {rest}");
        }
        let qualification = match (modal, group(&caps, "adverb")) {
            (Some("can"), _) => "can",
            (Some(_), _) => "possible",
            (None, Some("directly" | "always" | "necessarily")) => "explicit_assertion",
            (None, _) => "assertion",
        };
        let key = (
            factor,
            group(&caps, "effect").unwrap_or("").to_owned(),
            group(&caps, "condition").unwrap_or("").to_owned(),
        );
        found.entry(key).or_default().insert(qualification);
    }
    found
}

pub fn changes_influence_qualification(source: &str, candidate: &str) -> bool {
    const CONFLICTS: [[&str; 2]; 3] = [
        ["assertion", "possible"],
        ["explicit_assertion", "possible"],
        ["can", "explicit_assertion"],
    ];
    let (before, after) = (
        extract_influence_qualifications(source),
        extract_influence_qualifications(candidate),
    );
    before.iter().any(|(key, left)| {
        after.get(key).is_some_and(|right| {
            let union: BTreeSet<&str> = left.union(right).copied().collect();
            left.len() == 1
                && right.len() == 1
                && CONFLICTS
                    .iter()
                    .any(|conflict| union == conflict.iter().copied().collect())
        })
    })
}

// ── Coordinated outcomes ──

static COORDINATED_OUTCOMES_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"{SENTENCE_START}in (?P<condition>[a-z][a-z '-]{{0,72}}), (?P<article>an?|the) (?P<subject>[a-z][a-z'-]*(?: [a-z][a-z'-]*){{0,4}}?) (?:(?P<modal>may|might|could|can) )?(?P<first>[a-z]+) and (?P<second>[a-z]+) (?P<outcome>[a-z][a-z '-]{{0,72}})(?=[.!?]|$)"
        ),
        Flags::NONE,
    )
});
py_regex!(
    CONDITION_EXAMPLE_RE,
    r"^(?:wet|damp|dry|cold|hot|warm) (?:conditions|environments|climates|settings) like "
);

fn singular_present(verb: &str) -> String {
    if verb.ends_with('y') && verb.len() > 1 && !verb[..verb.len() - 1].ends_with(['a', 'e', 'i', 'o', 'u']) {
        return format!("{}ies", &verb[..verb.len() - 1]);
    }
    if verb.ends_with('s')
        || verb.ends_with("sh")
        || verb.ends_with("ch")
        || verb.ends_with('x')
        || verb.ends_with('z')
    {
        format!("{verb}es")
    } else {
        format!("{verb}s")
    }
}

pub type CoordinatedOutcomes = BTreeMap<(String, String, String, String, String), BTreeSet<&'static str>>;

/// Two matching predicates bound to an explicit condition and singular subject.
pub fn extract_coordinated_outcomes(text: &str) -> CoordinatedOutcomes {
    let prose = compact_folded(text);
    let mut outcomes = CoordinatedOutcomes::new();
    for caps in COORDINATED_OUTCOMES_RE.captures_iter(&prose) {
        let condition = CONDITION_EXAMPLE_RE
            .sub("", group(&caps, "condition").unwrap_or(""))
            .into_owned();
        let mut subject = group(&caps, "subject").unwrap_or("").to_owned();
        let start = whole(&caps).start;
        if group(&caps, "article") == Some("the") && !subject.contains(' ') && start > 0 {
            let prior_end = start - 1;
            let prior_start = ['.', '!', '?', '\n']
                .iter()
                .filter_map(|stop| prose[..prior_end].rfind(*stop))
                .max()
                .map_or(0, |at| at + 1);
            let prior = &prose[prior_start..prior_end];
            let head = format!(r"\b{}\b", escape(&subject));
            if cached_regex(&head, Flags::NONE).count(prior) == 1 {
                let mention = cached_regex(
                    &format!(r"\b(?:a|an|the) ((?:[a-z'-]+ ){{1,3}}{head})"),
                    Flags::NONE,
                );
                if let Some(found) = mention
                    .captures(prior)
                    .and_then(|caps| caps.get(1).map(|m| m.as_str().to_owned()))
                {
                    subject = found;
                }
            }
        }
        let modal = group(&caps, "modal");
        let (mut first, mut second) = (
            group(&caps, "first").unwrap_or("").to_owned(),
            group(&caps, "second").unwrap_or("").to_owned(),
        );
        if modal.is_some() {
            first = singular_present(&first);
            second = singular_present(&second);
        }
        let strength = match modal {
            None => "assertion",
            Some("can") => "ability",
            Some(_) => "possible",
        };
        let key = (
            condition,
            subject,
            first,
            second,
            group(&caps, "outcome").unwrap_or("").to_owned(),
        );
        outcomes.entry(key).or_default().insert(strength);
    }
    outcomes
}

pub fn changes_coordinated_outcome_qualification(source: &str, candidate: &str) -> bool {
    let (before, after) = (
        extract_coordinated_outcomes(source),
        extract_coordinated_outcomes(candidate),
    );
    before.iter().any(|(key, left)| {
        after.get(key).is_some_and(|right| {
            left.len() == 1
                && right.len() == 1
                && left.union(right).copied().collect::<BTreeSet<_>>()
                    == BTreeSet::from(["assertion", "possible"])
        })
    })
}
