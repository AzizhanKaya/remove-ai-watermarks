//! Optional English loan-role checks over caller-supplied dependency parses (`text_roles.py`).
//!
//! No model is loaded here. A `true` result means only that the selected parsed borrow/lend/loan
//! roles show no known reversal; it does not establish semantic equivalence.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::TextError;
use crate::unicode;

/// One token of a dependency parse (spaCy's `lemma_`, `pos_`, `dep_` and head index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToken {
    pub lemma: String,
    pub pos: String,
    pub dep: String,
    pub head: usize,
}

/// A complete parse of one input text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDoc {
    pub text: String,
    pub tokens: Vec<ParsedToken>,
    /// Whether dependency, POS and lemma annotations are complete for every token.
    pub complete: bool,
}

impl ParsedDoc {
    /// Parse the fixture notation `lemma/POS/dep/head` separated by whitespace.
    pub fn from_annotation(text: &str, annotation: &str) -> Result<Self, TextError> {
        let tokens = unicode::split_whitespace(annotation)
            .map(|row| {
                let fields: Vec<&str> = row.split('/').collect();
                let [lemma, pos, dep, head] = fields.as_slice() else {
                    return Err(TextError::invalid(format!("malformed token annotation {row:?}")));
                };
                let head = head
                    .parse()
                    .map_err(|_| TextError::invalid(format!("malformed head index in {row:?}")))?;
                Ok(ParsedToken {
                    lemma: (*lemma).to_owned(),
                    pos: (*pos).to_owned(),
                    dep: (*dep).to_owned(),
                    head,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            text: text.to_owned(),
            tokens,
            complete: true,
        })
    }

    fn children(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        self.tokens
            .iter()
            .enumerate()
            .filter(move |(at, token)| token.head == index && *at != index)
            .map(|(at, _)| at)
    }

    fn head(&self, index: usize) -> usize {
        self.tokens[index].head.min(self.tokens.len() - 1)
    }

    /// spaCy `Token.conjuncts`: every token linked by `conj`, excluding `index` itself.
    fn conjuncts(&self, index: usize) -> Vec<usize> {
        let mut root = index;
        let mut guard = 0;
        while self.tokens[root].dep == "conj" && guard < self.tokens.len() {
            root = self.head(root);
            guard += 1;
        }
        let mut pending = vec![root];
        let mut seen = BTreeSet::new();
        let mut result = Vec::new();
        while let Some(token) = pending.pop() {
            if !seen.insert(token) {
                continue;
            }
            if token != index {
                result.push(token);
            }
            pending.extend(
                self.children(token)
                    .filter(|&child| self.tokens[child].dep == "conj"),
            );
        }
        result
    }
}

/// A caller-owned English dependency parser.
pub trait DependencyParser {
    /// ISO language code of the pipeline.
    fn lang(&self) -> &str;
    /// Parse each text, in order.
    fn parse(&self, texts: &[&str]) -> Result<Vec<ParsedDoc>, TextError>;
}

const NOMINAL: [&str; 3] = ["NOUN", "PROPN", "PRON"];
const SUBJECT: [&str; 3] = ["nsubj", "nsubjpass", "nsubj:pass"];
const PASSIVE_SUBJECT: [&str; 2] = ["nsubjpass", "nsubj:pass"];
const OBJECT: [&str; 2] = ["dobj", "obj"];
const CONTROL: [&str; 12] = [
    "happy",
    "willing",
    "ready",
    "eager",
    "reluctant",
    "able",
    "want",
    "plan",
    "decide",
    "try",
    "hope",
    "agree",
];
const UNRESOLVED_PRONOUN: [&str; 15] = [
    "i", "me", "we", "us", "you", "he", "him", "she", "her", "they", "them", "it", "what", "which", "that",
];

type RoleKey = (bool, Option<String>, String);
type Roles = BTreeMap<RoleKey, BTreeSet<&'static str>>;

struct RoleReader<'d> {
    doc: &'d ParsedDoc,
}

impl RoleReader<'_> {
    fn token(&self, index: usize) -> &ParsedToken {
        &self.doc.tokens[index]
    }

    fn nominals(&self, index: usize) -> Vec<usize> {
        let mut found = vec![index];
        found.extend(
            self.doc
                .conjuncts(index)
                .into_iter()
                .filter(|&at| NOMINAL.contains(&self.token(at).pos.as_str())),
        );
        found
    }

    fn identity(&self, index: usize) -> Option<String> {
        let token = self.token(index);
        let lemma = unicode::casefold(&token.lemma);
        if !NOMINAL.contains(&token.pos.as_str()) || UNRESOLVED_PRONOUN.contains(&lemma.as_str()) {
            return None;
        }
        let mut parts: Vec<String> = self
            .doc
            .children(index)
            .filter(|&child| child < index && self.token(child).dep == "compound")
            .map(|child| unicode::casefold(&self.token(child).lemma))
            .collect();
        parts.push(lemma);
        Some(parts.join(" "))
    }

    fn subjects(&self, index: usize) -> Vec<usize> {
        let mut visited = BTreeSet::new();
        let mut current = index;
        while visited.insert(current) {
            let found: Vec<usize> = self
                .doc
                .children(current)
                .filter(|&child| SUBJECT.contains(&self.token(child).dep.as_str()))
                .collect();
            if !found.is_empty() {
                return found;
            }
            let token = self.token(current);
            let head = self.doc.head(current);
            let head_lemma = unicode::casefold(&self.token(head).lemma);
            let lemma = unicode::casefold(&token.lemma);
            let follow = token.dep == "conj"
                || (token.dep == "xcomp" && CONTROL.contains(&head_lemma.as_str()))
                || (token.dep == "acomp"
                    && CONTROL.contains(&lemma.as_str())
                    && self.token(head).lemma == "be");
            if !follow {
                break;
            }
            current = head;
        }
        Vec::new()
    }

    fn add(
        &self,
        roles: &mut Roles,
        negated: bool,
        items: &BTreeSet<String>,
        entity: usize,
        role: &'static str,
    ) {
        for nominal in self.nominals(entity) {
            let Some(name) = self.identity(nominal) else {
                continue;
            };
            roles
                .entry((negated, None, name.clone()))
                .or_default()
                .insert(role);
            for item in items {
                roles
                    .entry((negated, Some(item.clone()), name.clone()))
                    .or_default()
                    .insert(role);
            }
        }
    }

    fn roles(&self) -> Roles {
        let mut roles = Roles::new();
        for (index, token) in self.doc.tokens.iter().enumerate() {
            let verb = unicode::casefold(&token.lemma);
            if token.pos != "VERB" || !matches!(verb.as_str(), "borrow" | "lend" | "loan") {
                continue;
            }
            let borrow = verb == "borrow";
            let children: Vec<usize> = self.doc.children(index).collect();
            let subjects = self.subjects(index);
            let objects: Vec<usize> = children
                .iter()
                .copied()
                .filter(|&child| OBJECT.contains(&self.token(child).dep.as_str()))
                .collect();
            // A recipient passive keeps its direct object; otherwise the passive subject is the item.
            let mut item_tokens = objects.clone();
            if borrow || objects.is_empty() {
                item_tokens.extend(
                    subjects
                        .iter()
                        .copied()
                        .filter(|&subject| PASSIVE_SUBJECT.contains(&self.token(subject).dep.as_str())),
                );
            }
            let items: BTreeSet<String> = item_tokens
                .iter()
                .flat_map(|&item| self.nominals(item))
                .filter_map(|nominal| self.identity(nominal))
                .collect();
            let negated = children.iter().any(|&child| self.token(child).dep == "neg");
            let agent_role = if borrow { "borrower" } else { "lender" };
            for &subject in &subjects {
                if PASSIVE_SUBJECT.contains(&self.token(subject).dep.as_str()) {
                    if !borrow && !objects.is_empty() {
                        self.add(&mut roles, negated, &items, subject, "borrower");
                    }
                } else {
                    self.add(&mut roles, negated, &items, subject, agent_role);
                }
            }
            // The parser can attach from/to to the direct object rather than the verb.
            let mut arguments = children.clone();
            for &object in &objects {
                arguments.extend(
                    self.doc
                        .children(object)
                        .filter(|&child| self.token(child).dep == "prep"),
                );
            }
            for argument in arguments {
                let child = self.token(argument);
                if matches!(child.dep.as_str(), "dative" | "iobj")
                    && NOMINAL.contains(&child.pos.as_str())
                    && !borrow
                {
                    self.add(&mut roles, negated, &items, argument, "borrower");
                }
                if !matches!(child.dep.as_str(), "prep" | "agent" | "dative") {
                    continue;
                }
                let role = match unicode::casefold(&child.lemma).as_str() {
                    "by" => agent_role,
                    "from" if borrow => "lender",
                    "to" if !borrow => "borrower",
                    _ => continue,
                };
                for object in self
                    .doc
                    .children(argument)
                    .filter(|&grandchild| self.token(grandchild).dep == "pobj")
                    .collect::<Vec<_>>()
                {
                    self.add(&mut roles, negated, &items, object, role);
                }
            }
        }
        roles
    }
}

fn loan_roles(doc: &ParsedDoc) -> Result<Roles, TextError> {
    if doc.tokens.is_empty() || !doc.complete {
        return Err(TextError::invalid(
            "Loan-role checks require complete dependency, POS and lemma annotations",
        ));
    }
    Ok(RoleReader { doc }.roles())
}

/// Reject selected parsed English borrower/lender reversals.
///
/// Parser errors propagate, so an unavailable parse never becomes acceptance.
pub fn verify_text_rewrite_loan_roles(
    source: &str,
    candidate: &str,
    parser: &dyn DependencyParser,
) -> Result<bool, TextError> {
    if unicode::strip(source).is_empty() || unicode::strip(candidate).is_empty() {
        return Err(TextError::invalid(
            "Loan-role checks require nonempty source and candidate prose",
        ));
    }
    if parser.lang() != "en" {
        return Err(TextError::invalid(
            "Loan-role checks require a caller-selected English parser",
        ));
    }
    let docs = parser.parse(&[source, candidate])?;
    if docs.len() != 2 {
        return Err(TextError::invalid(
            "Loan-role parser must return one complete document per input",
        ));
    }
    if docs[0].text != source || docs[1].text != candidate {
        return Err(TextError::invalid(
            "Loan-role parser must preserve each exact input text",
        ));
    }
    let (before, after) = (loan_roles(&docs[0])?, loan_roles(&docs[1])?);
    Ok(!before
        .iter()
        .any(|(key, left)| after.get(key).is_some_and(|right| left.is_disjoint(right))))
}
