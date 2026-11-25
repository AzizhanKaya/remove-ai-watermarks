//! Quantities: numerals, English/Russian cardinals, measurements, frequencies, durations and bounds.

use std::fmt::Write as _;
use std::sync::LazyLock;

use fancy_regex::Captures;
use indexmap::IndexMap;

use crate::pyre::{Flags, PyRegex, Span, escape, group, group_span, whole};
use crate::unicode;

pub(crate) const PLACEHOLDER_TOKEN: &str = r"\[\[RAIW_SPAN_\d+\]\]";
pub(crate) const PLACEHOLDER_LIKE: &str = r"\[\[RAIW_SPAN_[^\]\r\n]*\]\]";
pub(crate) const NUMERAL_TOKEN: &str = r"(?<![\w.])[+\-−]?(?:\d[\d,.]*\d|\d|\.\d+)(?:[eE][+\-−]?\d+)?(?!\w)";

pub(crate) static PLACEHOLDER_TOKEN_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(PLACEHOLDER_TOKEN, Flags::NONE));
pub(crate) static PLACEHOLDER_LIKE_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(PLACEHOLDER_LIKE, Flags::NONE));
pub(crate) static NUMERAL_TOKEN_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(NUMERAL_TOKEN, Flags::NONE));

/// `_PLACEHOLDER_TOKEN_RE.sub(" ", text)`.
pub(crate) fn without_placeholders(text: &str) -> String {
    if !text.contains("[[RAIW_SPAN_") {
        return text.to_owned();
    }
    PLACEHOLDER_TOKEN_RE.sub(" ", text).into_owned()
}

pub(crate) const WORD_NUMBERS: [(&str, u64); 32] = [
    ("zero", 0),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
    ("hundred", 100),
    ("thousand", 1000),
    ("million", 1_000_000),
    ("billion", 1_000_000_000),
];

pub(crate) const ENGLISH_NUMBER_JOIN: &str = r"(?:[^\S\r\n]+|-)";
const JOIN: &str = ENGLISH_NUMBER_JOIN;

fn word_group(filter: impl Fn(u64) -> bool) -> String {
    let words: Vec<&str> = WORD_NUMBERS
        .iter()
        .filter(|(_, value)| filter(*value))
        .map(|(word, _)| *word)
        .collect();
    format!("(?:{})", words.join("|"))
}

/// Python `sorted(words, key=len, reverse=True)`: stable, longest first.
fn longest_first<'a>(words: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut words: Vec<&str> = words.into_iter().collect();
    words.sort_by_key(|word| std::cmp::Reverse(word.chars().count()));
    words
}

pub(crate) static ENGLISH_INTEGER: LazyLock<String> = LazyLock::new(|| {
    let small = word_group(|value| value < 20);
    let unit = word_group(|value| 0 < value && value < 10);
    let tens = word_group(|value| (20..100).contains(&value));
    let below_hundred = format!("(?:{tens}(?:{JOIN}{unit})?|{small})");
    let below_thousand = format!(
        "(?:(?:(?:{unit}|a|an){JOIN})?hundred(?:{JOIN}(?:and{JOIN})?{below_hundred})?|{below_hundred})"
    );
    let scale_amount = format!("(?:(?:{below_thousand}|a|an){JOIN})?");
    let parts = [
        format!("{scale_amount}billion"),
        format!("{scale_amount}million"),
        format!("{scale_amount}thousand"),
        below_thousand,
    ];
    let heads: Vec<String> = parts
        .iter()
        .enumerate()
        .map(|(index, head)| {
            parts[index + 1..].iter().fold(head.clone(), |mut acc, tail| {
                let _ = write!(acc, "(?:{JOIN}(?:and{JOIN})?{tail})?");
                acc
            })
        })
        .collect();
    format!(
        r"(?<!\w)(?:(?:minus|plus){JOIN})?(?:{})\b(?!{JOIN}(?:hundred|thousand|million|billion)\b)",
        heads.join("|")
    )
});
pub(crate) static ENGLISH_INTEGER_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&ENGLISH_INTEGER, Flags::I));

pub(crate) static ENGLISH_WORD_NUMERAL: LazyLock<String> = LazyLock::new(|| {
    let word = format!(
        "(?:{})",
        longest_first(WORD_NUMBERS.iter().map(|(word, _)| *word)).join("|")
    );
    let article = format!(r"(?:(?:a|an){JOIN}(?=(?:hundred|thousand|million|billion)\b))?");
    format!(r"\b(?:(?:minus|plus){JOIN})?{article}{word}(?:{JOIN}(?:and{JOIN})?{article}{word})*\b")
});
pub(crate) static ENGLISH_WORD_NUMERAL_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&ENGLISH_WORD_NUMERAL, Flags::I));

// ── Frequencies ──

const FREQUENCY_GAP: &str = r"[^\S\r\n]+";
const FREQUENCY_UNIT: &str = r"(?:hour|day|week|month|year)s?";
const FREQUENCY_ADVERBS: [(&str, &str); 7] = [
    ("hourly", "hour"),
    ("daily", "day"),
    ("weekly", "week"),
    ("monthly", "month"),
    ("yearly", "year"),
    ("annually", "year"),
    ("annual", "year"),
];

static FREQUENCY_REPEAT: LazyLock<String> = LazyLock::new(|| {
    let amount = format!(r"(?:[1-9]\d*|(?!(?:minus|zero)\b){})", *ENGLISH_INTEGER);
    format!("(?:once|twice|thrice|{amount}{FREQUENCY_GAP}times)")
});

pub(crate) static FREQUENCY_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let amount = format!(r"(?:[1-9]\d*|(?!(?:minus|zero)\b){})", *ENGLISH_INTEGER);
    let adverbs: Vec<&str> = FREQUENCY_ADVERBS.iter().map(|(adverb, _)| *adverb).collect();
    let gap = FREQUENCY_GAP;
    let unit = FREQUENCY_UNIT;
    let interval = format!(
        "(?:(?:every|each){gap}(?:(?:other|second|{amount}){gap})?{unit}|on{gap}alternate{gap}{unit}|per{gap}{unit}|(?:{}))",
        adverbs.join("|")
    );
    format!(
        r"(?<![\w-])(?:{}{gap}(?:(?:a|an|per){gap}{unit}|{interval})|{interval})\b",
        *FREQUENCY_REPEAT
    )
});
pub(crate) static FREQUENCY_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&FREQUENCY_PATTERN, Flags::I));
static FREQUENCY_REPEAT_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&format!("^({}){FREQUENCY_GAP}", *FREQUENCY_REPEAT), Flags::I));
static FREQUENCY_UNIT_END_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&format!("({FREQUENCY_UNIT})$"), Flags::NONE));

/// A recurrence signature `(repeat_count, period, unit)`.
pub type Frequency = (String, String, String);

/// Selected explicit English recurrence counts, periods and multiplicity.
pub fn extract_frequencies(text: &str) -> Vec<Frequency> {
    let prose = without_placeholders(text);
    let mut frequencies = Vec::new();
    for span in FREQUENCY_RE.find_iter(&prose) {
        let mut phrase = unicode::fold_words(&prose[span.range()]);
        let mut count = "1".to_owned();
        if let Some(repeat) = FREQUENCY_REPEAT_RE.captures(&phrase) {
            let token = repeat.get(1).map_or("", |m| m.as_str());
            count = match token {
                "once" => "1".into(),
                "twice" => "2".into(),
                "thrice" => "3".into(),
                _ => normalize_measurement_amount(token.strip_suffix(" times").unwrap_or(token)),
            };
            let end = whole(&repeat).end;
            phrase = phrase[end..].to_owned();
        }
        let (unit, period) =
            if let Some((_, unit)) = FREQUENCY_ADVERBS.iter().find(|(adverb, _)| *adverb == phrase) {
                ((*unit).to_owned(), "1".to_owned())
            } else {
                let literal = FREQUENCY_UNIT_END_RE
                    .findall(&phrase)
                    .first()
                    .map(|unit| (*unit).to_owned())
                    .unwrap_or_default();
                let unit = literal.strip_suffix('s').unwrap_or(&literal).to_owned();
                let prefix = unicode::strip(&phrase[..phrase.len() - literal.len()]).to_owned();
                let period = match prefix.as_str() {
                    "on alternate" => "2".to_owned(),
                    "a" | "an" | "per" => "1".to_owned(),
                    _ => {
                        let amount = prefix.split_once(' ').map_or("", |(_, rest)| rest);
                        match amount {
                            "other" | "second" => "2".to_owned(),
                            "" => "1".to_owned(),
                            _ => normalize_measurement_amount(amount),
                        }
                    }
                };
                (unit, period)
            };
        frequencies.push((count, period, unit));
    }
    frequencies.sort();
    frequencies
}

/// Read a grammar-checked English integer without summing unrelated number words.
pub fn normalize_english_integer(token: &str) -> String {
    if !ENGLISH_INTEGER_RE.fullmatch(token) {
        return format!("en-unparsed:{}", unicode::fold_words(token));
    }
    let folded = unicode::casefold(token);
    let words: Vec<&str> = folded
        .split(|ch: char| !ch.is_ascii_lowercase())
        .filter(|word| !word.is_empty())
        .collect();
    let sign: i128 = if words.first() == Some(&"minus") { -1 } else { 1 };
    let mut total: i128 = 0;
    let mut current: Option<i128> = None;
    for word in words {
        if matches!(word, "minus" | "plus" | "and" | "a" | "an") {
            continue;
        }
        let Some(&(_, value)) = WORD_NUMBERS.iter().find(|(name, _)| *name == word) else {
            continue;
        };
        let value = i128::from(value);
        if value == 100 {
            current = Some(current.unwrap_or(1) * value);
        } else if value >= 1000 {
            total += current.unwrap_or(1) * value;
            current = None;
        } else {
            current = Some(current.unwrap_or(0) + value);
        }
    }
    (sign * (total + current.unwrap_or(0))).to_string()
}

/// Split conjunction lists after giving a complete scalar first priority.
fn english_number_pieces(text: &str, span: Span) -> Vec<(usize, usize, String)> {
    static CONJUNCTION_RE: LazyLock<PyRegex> =
        LazyLock::new(|| PyRegex::new(&format!("{JOIN}and{JOIN}"), Flags::I));
    let raw = &text[span.range()];
    let unparsed = || {
        vec![(
            span.start,
            span.end,
            format!("en-unparsed:{}", unicode::fold_words(raw)),
        )]
    };
    let mut pieces = Vec::new();
    let mut cursor = 0;
    while cursor < raw.len() {
        let Some(scalar) = ENGLISH_INTEGER_RE.match_at(raw, cursor) else {
            return unparsed();
        };
        pieces.push((
            span.start + cursor,
            span.start + scalar.end,
            normalize_english_integer(&raw[scalar.range()]),
        ));
        cursor = scalar.end;
        if cursor == raw.len() {
            return pieces;
        }
        let Some(conjunction) = CONJUNCTION_RE.match_at(&raw[cursor..], 0) else {
            return unparsed();
        };
        cursor += conjunction.end;
    }
    pieces
}

// ── Russian cardinals ──

const RUSSIAN_CARDINAL_GROUPS: [(u64, &str); 37] = [
    (0, "ноль нуль нуля нулю нулем нулём нуле"),
    (1, "один одна одно одну одного одной одному одним одном"),
    (2, "два две двух двум двумя"),
    (3, "три трех трёх трем трём тремя"),
    (4, "четыре четырех четырёх четырем четырём четырьмя"),
    (5, "пять пяти пятью"),
    (6, "шесть шести шестью"),
    (7, "семь семи семью"),
    (8, "восемь восьми восемью восьмью"),
    (9, "девять девяти девятью"),
    (10, "десять десяти десятью"),
    (11, "одиннадцать одиннадцати одиннадцатью"),
    (12, "двенадцать двенадцати двенадцатью"),
    (13, "тринадцать тринадцати тринадцатью"),
    (14, "четырнадцать четырнадцати четырнадцатью"),
    (15, "пятнадцать пятнадцати пятнадцатью"),
    (16, "шестнадцать шестнадцати шестнадцатью"),
    (17, "семнадцать семнадцати семнадцатью"),
    (18, "восемнадцать восемнадцати восемнадцатью"),
    (19, "девятнадцать девятнадцати девятнадцатью"),
    (20, "двадцать двадцати двадцатью"),
    (30, "тридцать тридцати тридцатью"),
    // The oblique 'сорока' is also the ordinary noun 'magpie'; leave it unparsed.
    (40, "сорок"),
    (50, "пятьдесят пятидесяти пятьюдесятью"),
    (60, "шестьдесят шестидесяти шестьюдесятью"),
    (70, "семьдесят семидесяти семьюдесятью"),
    (80, "восемьдесят восьмидесяти восемьюдесятью"),
    (90, "девяносто девяноста"),
    (100, "сто ста"),
    (200, "двести двухсот двумстам двумястами двухстах"),
    (
        300,
        "триста трехсот трёхсот тремстам трёмстам тремястами трехстах трёхстах",
    ),
    (
        400,
        "четыреста четырехсот четырёхсот четыремстам четырёмстам четырьмястами четырехстах четырёхстах",
    ),
    (500, "пятьсот пятисот пятистам пятьюстами пятистах"),
    (600, "шестьсот шестисот шестистам шестьюстами шестистах"),
    (700, "семьсот семисот семистам семьюстами семистах"),
    (
        800,
        "восемьсот восьмисот восьмистам восемьюстами восьмьюстами восьмистах",
    ),
    (900, "девятьсот девятисот девятистам девятьюстами девятистах"),
];
const RUSSIAN_SCALE_GROUPS: [(u64, &str); 3] = [
    (
        1000,
        "тысяча тысячи тысяч тысячу тысяче тысячей тысячам тысячами тысячах",
    ),
    (
        1_000_000,
        "миллион миллионы миллиона миллионов миллиону миллионом миллионе миллионам миллионами миллионах",
    ),
    (
        1_000_000_000,
        "миллиард миллиарды миллиарда миллиардов миллиарду миллиардом миллиарде миллиардам миллиардами миллиардах",
    ),
];

fn word_values(groups: &[(u64, &'static str)]) -> IndexMap<&'static str, u64> {
    let mut map = IndexMap::new();
    for &(value, forms) in groups {
        for word in forms.split(' ') {
            map.insert(word, value);
        }
    }
    map
}

static RUSSIAN_CARDINAL_VALUES: LazyLock<IndexMap<&'static str, u64>> =
    LazyLock::new(|| word_values(&RUSSIAN_CARDINAL_GROUPS));
static RUSSIAN_SCALE_VALUES: LazyLock<IndexMap<&'static str, u64>> =
    LazyLock::new(|| word_values(&RUSSIAN_SCALE_GROUPS));

pub(crate) static RUSSIAN_NUMBER_WORD_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let words = longest_first(
        RUSSIAN_CARDINAL_VALUES
            .keys()
            .chain(RUSSIAN_SCALE_VALUES.keys())
            .copied(),
    );
    format!("(?:{})", words.join("|"))
});
pub(crate) static RUSSIAN_NUMERAL: LazyLock<String> = LazyLock::new(|| {
    let word = &*RUSSIAN_NUMBER_WORD_PATTERN;
    format!(r"\b(?:(?:минус|плюс)[^\S\r\n]+)?{word}(?:[^\S\r\n]+{word})*\b")
});
pub(crate) static RUSSIAN_NUMERAL_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&RUSSIAN_NUMERAL, Flags::I));
pub(crate) static RUSSIAN_DURATION: LazyLock<String> = LazyLock::new(|| {
    format!(
        r"\b(?P<number_prefix>{}+)(?P<duration_unit>летн|дневн|часов|недельн|месячн|минутн|секундн)(?:ий|ый|ой|ая|яя|ое|ее|ые|ие|ого|его|ей|ых|их|ым|им|ыми|ими|ом|ем|ому|ему|ую|юю)\b",
        *RUSSIAN_NUMBER_WORD_PATTERN
    )
});
pub(crate) static RUSSIAN_DURATION_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&RUSSIAN_DURATION, Flags::I));
static RUSSIAN_NUMBER_WORD_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(&RUSSIAN_NUMBER_WORD_PATTERN, Flags::NONE));

/// Normalize selected Russian cardinals; unparsed sequences stay literal facts.
pub fn normalize_russian_integer(token: &str) -> String {
    let folded = unicode::casefold(token);
    let mut words: Vec<&str> = unicode::split_whitespace(&folded).collect();
    let unparsed = format!("ru-unparsed:{}", words.join(" "));
    let sign: i128 = if words.first() == Some(&"минус") {
        -1
    } else {
        1
    };
    if matches!(words.first(), Some(&("минус" | "плюс"))) {
        words.remove(0);
    }
    let cardinal = |word: &str| RUSSIAN_CARDINAL_VALUES.get(word).map(|&value| i128::from(value));
    let mut total: i128 = 0;
    let mut previous_scale: i128 = 1_000_000_000_000;
    let mut index = 0;
    while index < words.len() {
        let start = index;
        let mut group: i128 = 0;
        if let Some(value) = cardinal(words[index]) {
            group = value;
            index += 1;
            if group >= 100
                && let Some(following) = words.get(index).and_then(|word| cardinal(word))
                && 0 < following
                && following < 100
            {
                group += following;
                index += 1;
            }
            if (20..=90).contains(&(group % 100))
                && group % 10 == 0
                && let Some(following) = words.get(index).and_then(|word| cardinal(word))
                && 0 < following
                && following < 10
            {
                group += following;
                index += 1;
            }
        }
        if let Some(&scale) = words.get(index).and_then(|word| RUSSIAN_SCALE_VALUES.get(word)) {
            let scale = i128::from(scale);
            if index == start {
                if !matches!(words[index], "тысяча" | "тысячу" | "миллион" | "миллиард")
                {
                    return unparsed;
                }
                group = 1;
            }
            if group == 0 || scale >= previous_scale {
                return unparsed;
            }
            total += group * scale;
            previous_scale = scale;
            index += 1;
        } else if index == words.len() && index > start {
            return (sign * (total + group)).to_string();
        } else {
            return unparsed;
        }
    }
    (sign * total).to_string()
}

/// Read a selected explicit number prefix without treating summer as a count.
fn normalize_russian_duration(caps: &Captures<'_, str>) -> String {
    let prefix = unicode::casefold(group(caps, "number_prefix").unwrap_or(""));
    let words = RUSSIAN_NUMBER_WORD_RE.findall(&prefix);
    if let [word] = words.as_slice()
        && let Some(&scale) = RUSSIAN_SCALE_VALUES.get(word)
    {
        return scale.to_string();
    }
    normalize_russian_integer(&words.join(" "))
}

fn russian_duration_unit(caps: &Captures<'_, str>) -> &'static str {
    match unicode::casefold(group(caps, "duration_unit").unwrap_or("")).as_str() {
        "летн" => "year",
        "дневн" => "day",
        "часов" => "hour",
        "недельн" => "week",
        "месячн" => "month",
        "минутн" => "minute",
        _ => "second",
    }
}

// ── Measurements ──

const MEASUREMENT_UNITS: [(&str, &str, &str); 34] = [
    (
        "km",
        "km|км",
        "kilometer|kilometers|kilometre|kilometres|километр|километра|километров|километры",
    ),
    ("m", "m|м", "meter|meters|metre|metres|метр|метра|метров|метры"),
    (
        "cm",
        "cm|см",
        "centimeter|centimeters|centimetre|centimetres|сантиметр|сантиметра|сантиметров",
    ),
    (
        "mm",
        "mm|мм",
        "millimeter|millimeters|millimetre|millimetres|миллиметр|миллиметра|миллиметров",
    ),
    ("mile", "", "mile|miles|миля|мили|миль"),
    ("foot", "ft", "foot|feet|фут|фута|футов"),
    ("inch", "", "inch|inches|дюйм|дюйма|дюймов"),
    (
        "kg",
        "kg|кг",
        "kilogram|kilograms|килограмм|килограмма|килограммов|килограммы",
    ),
    ("g", "g", "gram|grams|грамм|грамма|граммов"),
    (
        "mg",
        "mg|мг",
        "milligram|milligrams|миллиграмм|миллиграмма|миллиграммов",
    ),
    (
        "ug",
        "μg|µg|мкг",
        "microgram|micrograms|микрограмм|микрограмма|микрограммов",
    ),
    ("lb", "lb|lbs", "pound|pounds|фунт|фунта|фунтов"),
    ("oz", "oz", "ounce|ounces|унция|унции|унций"),
    ("L", "L|l|л", "liter|liters|litre|litres|литр|литра|литров"),
    (
        "mL",
        "mL|ml|мл",
        "milliliter|milliliters|millilitre|millilitres|миллилитр|миллилитра|миллилитров",
    ),
    ("ML", "ML", "megaliter|megaliters|megalitre|megalitres"),
    ("W", "W|Вт", "watt|watts|ватт|ватта|ваттов"),
    (
        "mW",
        "mW|мВт",
        "milliwatt|milliwatts|милливатт|милливатта|милливаттов",
    ),
    ("kW", "kW|кВт", "kilowatt|kilowatts|киловатт|киловатта|киловаттов"),
    ("MW", "MW|МВт", "megawatt|megawatts|мегаватт|мегаватта|мегаваттов"),
    ("Celsius", "°C", "degree Celsius|degrees Celsius"),
    ("Fahrenheit", "°F", "degree Fahrenheit|degrees Fahrenheit"),
    ("kelvin", "K", "kelvin|kelvins|кельвин|кельвина|кельвинов"),
    ("second", "s|sec", "second|seconds|секунда|секунды|секунд"),
    (
        "millisecond",
        "ms",
        "millisecond|milliseconds|миллисекунда|миллисекунды|миллисекунд",
    ),
    ("minute", "min", "minute|minutes|минута|минуты|минут"),
    ("hour", "h|hr", "hour|hours|час|часа|часов"),
    ("day", "d", "day|days|день|дня|дней"),
    ("week", "", "week|weeks|неделя|недели|недель"),
    ("month", "", "month|months|месяц|месяца|месяцев"),
    ("year", "", "year|years|год|года|лет"),
    ("percent", "%", "percent|процент|процента|процентов"),
    (
        "m2",
        "m²|m^2",
        "square meter|square meters|square metre|square metres",
    ),
    (
        "cm2",
        "cm²|cm^2",
        "square centimeter|square centimeters|square centimetre|square centimetres",
    ),
];
const KM2_UNIT: (&str, &str, &str) = (
    "km2",
    "km²|km^2",
    "square kilometer|square kilometers|square kilometre|square kilometres",
);

fn measurement_units() -> impl Iterator<Item = (&'static str, &'static str, &'static str)> {
    MEASUREMENT_UNITS.into_iter().chain(std::iter::once(KM2_UNIT))
}

static MEASUREMENT_SYMBOLS: LazyLock<IndexMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut map = IndexMap::new();
    for (key, symbols, _) in measurement_units() {
        for alias in symbols.split('|').filter(|alias| !alias.is_empty()) {
            map.insert(alias, key);
        }
    }
    map
});
static MEASUREMENT_WORDS: LazyLock<IndexMap<String, &'static str>> = LazyLock::new(|| {
    let mut map = IndexMap::new();
    for (key, _, words) in measurement_units() {
        for alias in words.split('|') {
            map.insert(unicode::casefold(alias), key);
        }
    }
    map
});

pub(crate) static MEASUREMENT_ATOM_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let symbols: Vec<String> = longest_first(MEASUREMENT_SYMBOLS.keys().copied())
        .into_iter()
        .map(escape)
        .collect();
    let words: Vec<String> = longest_first(MEASUREMENT_WORDS.keys().map(String::as_str))
        .into_iter()
        .map(|word| {
            word.split(' ')
                .map(escape)
                .collect::<Vec<_>>()
                .join(r"[^\S\r\n]+")
        })
        .collect();
    format!("(?:(?-i:{})|(?:{}))", symbols.join("|"), words.join("|"))
});
pub(crate) static MEASUREMENT_AMOUNT_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let numeral = NUMERAL_TOKEN.strip_suffix(r"(?!\w)").unwrap_or(NUMERAL_TOKEN);
    format!("(?:{}|{}|{numeral})", *RUSSIAN_NUMERAL, *ENGLISH_WORD_NUMERAL)
});
pub(crate) static MEASUREMENT_UNIT_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let atom = &*MEASUREMENT_ATOM_PATTERN;
    format!(r"{atom}(?:[^\S\r\n]*/[^\S\r\n]*{atom})*")
});
pub(crate) const MEASUREMENT_END: &str = r"(?!\w|[^\S\r\n]*[/^*·])";
pub(crate) const MEASUREMENT_GAP: &str = r"(?:[^\S\r\n]*|-(?=(?:hour|day|week|month|year)\b))";
pub(crate) static MEASUREMENT_PATTERN: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{MEASUREMENT_GAP}{}{MEASUREMENT_END}",
        *MEASUREMENT_AMOUNT_PATTERN, *MEASUREMENT_UNIT_PATTERN
    )
});
pub(crate) static MEASUREMENT_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            "(?P<amount>{}){MEASUREMENT_GAP}(?P<unit>{}){MEASUREMENT_END}",
            *MEASUREMENT_AMOUNT_PATTERN, *MEASUREMENT_UNIT_PATTERN
        ),
        Flags::I,
    )
});
pub(crate) static PERCENT_CHANGE_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"\b(?:(?:increase|decrease|reduction|rise|fall|drop|decline|growth)(?:[ \t]+in(?:[ \t]+(?!(?:to|by|of|from)\b)[A-Za-z][A-Za-z-]*){{1,6}})?[ \t]+(?P<noun_link>to|of|by)|(?:rose|risen|grew|grown|fell|fallen|increased|decreased|dropped|declined|reduced)[ \t]+(?P<verb_link>to|by))[ \t]+(?P<amount>{})[^\S\r\n]*(?P<unit>%|percent){MEASUREMENT_END}",
            *MEASUREMENT_AMOUNT_PATTERN
        ),
        Flags::I,
    )
});
static UNIT_SPLIT_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(r"[^\S\r\n]*/[^\S\r\n]*", Flags::NONE));

pub(crate) const INDEFINITE_DURATION_PATTERN: &str =
    r"\b(?:a|an)[^\S\r\n]+(?:hour|day|week|month|year)\b(?![\w-]|[^\S\r\n]*[/^*·])";
pub(crate) static INDEFINITE_DURATION_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(INDEFINITE_DURATION_PATTERN, Flags::I));
pub(crate) const VAGUE_DURATION_PATTERN: &str = r"\bfor[^\S\r\n]+(?:(?:many|several|a[^\S\r\n]+few)[^\S\r\n]+)?(?:seconds|minutes|hours|days|weeks|months|years|decades|centuries)\b(?![\w-])";
pub(crate) static VAGUE_DURATION_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(VAGUE_DURATION_PATTERN, Flags::I));
// Limit matches to adverbial phrases; "on the weekend shift" can name a role.
pub(crate) const WEEKEND_TIMING_PATTERN: &str = r"\b(?:(?P<whole_weekend>(?:all|throughout[ \t]+the)[ \t]+weekend)|(?P<relative_weekend>last|this|next)[ \t]+weekend|(?P<recurring_weekend>(?:(?:every|each)[ \t]+weekend|(?:on|at|during|over)[ \t]+weekends))|(?P<single_weekend>(?:on|at|during|over)[ \t]+the[ \t]+weekend))(?=[ \t]*(?:[,.;!?\r\n]|\Z)|[ \t]+(?:before|after)\b)";
pub(crate) static WEEKEND_TIMING_RE: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(WEEKEND_TIMING_PATTERN, Flags::I));

pub fn extract_vague_durations(text: &str) -> Vec<String> {
    let mut found: Vec<String> = VAGUE_DURATION_RE
        .find_iter(text)
        .map(|span| unicode::fold_words(&text[span.range()]))
        .collect();
    found.sort();
    found
}

pub fn extract_weekend_timing(text: &str) -> Vec<String> {
    const KINDS: [&str; 4] = [
        "whole_weekend",
        "relative_weekend",
        "recurring_weekend",
        "single_weekend",
    ];
    let mut timing: Vec<String> = WEEKEND_TIMING_RE
        .captures_iter(text)
        .filter_map(|caps| {
            let kind = crate::pyre::last_group(&caps, &KINDS)?;
            Some(if kind == "relative_weekend" {
                unicode::casefold(group(&caps, kind)?)
            } else {
                kind.to_owned()
            })
        })
        .collect();
    timing.sort();
    timing
}

pub(crate) fn normalize_measurement_amount(amount: &str) -> String {
    if RUSSIAN_NUMERAL_RE.fullmatch(amount) {
        return normalize_russian_integer(amount);
    }
    if ENGLISH_WORD_NUMERAL_RE.fullmatch(amount) {
        return normalize_english_integer(amount);
    }
    normalize_numeral_token(amount)
}

/// A measurement `(amount, unit)` signature.
pub type Measurement = (String, String);

fn measurement_signature(caps: &Captures<'_, str>) -> Measurement {
    let unit_text = group(caps, "unit").unwrap_or("");
    let units: Vec<&str> = UNIT_SPLIT_RE
        .split_keep(unit_text)
        .into_iter()
        .map(|atom| {
            MEASUREMENT_SYMBOLS.get(atom).copied().unwrap_or_else(|| {
                MEASUREMENT_WORDS
                    .get(&unicode::fold_words(atom))
                    .copied()
                    .unwrap_or("")
            })
        })
        .collect();
    (
        normalize_measurement_amount(group(caps, "amount").unwrap_or("")),
        units.join("/"),
    )
}

fn indefinite_duration_signature(text: &str) -> Measurement {
    let folded = unicode::casefold(text);
    (
        "1".to_owned(),
        unicode::split_whitespace(&folded).last().unwrap_or("").to_owned(),
    )
}

fn duration_measurement_signature(caps: &Captures<'_, str>) -> Measurement {
    (
        normalize_russian_duration(caps),
        russian_duration_unit(caps).to_owned(),
    )
}

/// Adjacent measurements with multiplicity; no conversion or role inference.
pub fn extract_measurements(text: &str) -> Vec<Measurement> {
    let prose = without_placeholders(text);
    let prose = FREQUENCY_RE.sub(" ", &prose);
    let mut measurements: Vec<Measurement> = MEASUREMENT_RE
        .captures_iter(&prose)
        .map(|caps| measurement_signature(&caps))
        .collect();
    measurements.extend(
        INDEFINITE_DURATION_RE
            .find_iter(&prose)
            .map(|span| indefinite_duration_signature(&prose[span.range()])),
    );
    measurements.extend(
        RUSSIAN_DURATION_RE
            .captures_iter(&prose)
            .map(|caps| duration_measurement_signature(&caps)),
    );
    measurements.sort();
    measurements
}

// ── Numeral tokens ──

/// `_normalize_numeral_token`: exact decimal digits, or a literal for ambiguous syntax.
pub fn normalize_numeral_token(token: &str) -> String {
    static GROUPED_RE: LazyLock<PyRegex> =
        LazyLock::new(|| PyRegex::new(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?", Flags::NONE));
    let cleaned = token.replace('\u{2212}', "-");
    let literal = || format!("literal-number:{}", unicode::casefold(&cleaned));
    let cleaned = if cleaned.contains(',') {
        if !GROUPED_RE.fullmatch(&cleaned) {
            return literal();
        }
        cleaned.replace(',', "")
    } else {
        cleaned.clone()
    };
    let Some(decimal) = parse_decimal(&cleaned) else {
        return literal();
    };
    let significant = decimal.digits;
    if significant.bytes().all(|digit| digit == b'0') {
        return "0".to_owned();
    }
    let stripped = significant.trim_end_matches('0');
    let exponent = decimal.exponent + (significant.len() - stripped.len()) as i128;
    let prefix = if decimal.negative { "-" } else { "" };
    let length = stripped.len() as i128;
    // Keep exact tuples for extreme magnitudes instead of expanding an exponent.
    if (length + exponent).abs() > 4096 {
        return format!("decimal:{prefix}{stripped}e{exponent}");
    }
    if exponent >= 0 {
        return format!("{prefix}{stripped}{}", "0".repeat(exponent as usize));
    }
    let point = length + exponent;
    if point > 0 {
        let point = point as usize;
        return format!("{prefix}{}.{}", &stripped[..point], &stripped[point..]);
    }
    format!("{prefix}0.{}{stripped}", "0".repeat((-point) as usize))
}

struct Decimal {
    negative: bool,
    digits: String,
    exponent: i128,
}

/// `decimal.Decimal(text).as_tuple()` for finite numerals, with libmpdec's exponent limits.
fn parse_decimal(text: &str) -> Option<Decimal> {
    const MAX_EMAX: i128 = 999_999_999_999_999_999;
    const MIN_ETINY: i128 = -1_999_999_999_999_999_997;
    let mut chars = text.chars().peekable();
    let negative = match chars.peek() {
        Some('-') => {
            chars.next();
            true
        }
        Some('+') => {
            chars.next();
            false
        }
        _ => false,
    };
    let mut digits = String::new();
    let mut fraction = 0i128;
    let mut seen_point = false;
    let mut seen_digit = false;
    let mut exponent: i128 = 0;
    while let Some(&ch) = chars.peek() {
        if let Some(value) = unicode::decimal_value(ch) {
            digits.push(char::from(b'0' + value as u8));
            seen_digit = true;
            if seen_point {
                fraction += 1;
            }
        } else if ch == '.' && !seen_point {
            seen_point = true;
        } else if ch == 'e' || ch == 'E' {
            chars.next();
            let mut sign = 1;
            match chars.peek() {
                Some('-') => {
                    sign = -1;
                    chars.next();
                }
                Some('+') => {
                    chars.next();
                }
                _ => {}
            }
            let mut exp_digits = 0;
            for ch in chars.by_ref() {
                let value = unicode::decimal_value(ch)?;
                exp_digits += 1;
                if exp_digits > 30 {
                    return None;
                }
                exponent = exponent * 10 + i128::from(value);
            }
            if exp_digits == 0 {
                return None;
            }
            exponent *= sign;
            break;
        } else {
            return None;
        }
        chars.next();
    }
    if !seen_digit {
        return None;
    }
    let trimmed = digits.trim_start_matches('0');
    let digits = if trimmed.is_empty() {
        "0".to_owned()
    } else {
        trimmed.to_owned()
    };
    let exponent = exponent - fraction;
    let adjusted = exponent + digits.len() as i128 - 1;
    if adjusted > MAX_EMAX || exponent < MIN_ETINY {
        return None;
    }
    Some(Decimal {
        negative,
        digits,
        exponent,
    })
}

// ── Non-quantity "one" ──

static NONQUANTITY_ONE_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    let ones: Vec<&str> = RUSSIAN_CARDINAL_VALUES
        .iter()
        .filter(|&(_, &value)| value == 1)
        .map(|(word, _)| *word)
        .collect();
    PyRegex::new(
        &format!(
            r"\b(?:one\s+(?:of\s+(?:the\s+)?(?:most|least|best|worst)\b|another\b)|(?:{})\s+из\s+(?:самых|лучших|худших)\b)",
            ones.join("|")
        ),
        Flags::I,
    )
});
static RECIPROCAL_ONE_RE: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::new(r"one\s+another\b", Flags::I));
static GENERIC_KNOWLEDGE_SUBJECT_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        r"(?:\A|(?<=[.!?]))\s*to (?:understand|comprehend|grasp|know) [^,.!?\n]{1,160},\s*(?P<subject>one) (?=must (?:first )?(?:understand|comprehend|grasp|know)\b)",
        Flags::I,
    )
});
static INDEFINITE_PARTICIPLE_ONE_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        r"\b(?:a|an)[^\S\r\n]+(?:partly|partially|fully|completely)[^\S\r\n]+[a-z]+ed[^\S\r\n]+(?P<substitute>one)(?=[,.;!?]|$)",
        Flags::I,
    )
});

pub(crate) fn contextual_one_offsets(text: &str) -> Vec<usize> {
    let mut offsets: Vec<usize> = GENERIC_KNOWLEDGE_SUBJECT_RE
        .captures_iter(text)
        .filter_map(|caps| group_span(&caps, "subject").map(|span| span.start))
        .collect();
    offsets.extend(
        INDEFINITE_PARTICIPLE_ONE_RE
            .captures_iter(text)
            .filter_map(|caps| group_span(&caps, "substitute").map(|span| span.start)),
    );
    offsets
}

pub(crate) fn is_nonquantity_one(text: &str, start: usize, contextual: &[usize]) -> bool {
    contextual.contains(&start) || NONQUANTITY_ONE_RE.match_at(text, start).is_some()
}

/// Sorted numerical-mention occurrences (`_extract_numeral_occurrences`).
pub fn extract_numeral_occurrences(text: &str) -> Vec<String> {
    let prose = without_placeholders(text);
    let prose = FREQUENCY_RE.sub(" ", &prose).into_owned();
    let mut numbers: Vec<String> = MEASUREMENT_RE
        .captures_iter(&prose)
        .map(|caps| normalize_measurement_amount(group(&caps, "amount").unwrap_or("")))
        .collect();
    let prose = MEASUREMENT_RE.sub(" ", &prose).into_owned();
    numbers.extend(INDEFINITE_DURATION_RE.find_iter(&prose).map(|_| "1".to_owned()));
    let prose = INDEFINITE_DURATION_RE.sub(" ", &prose).into_owned();
    numbers.extend(
        NUMERAL_TOKEN_RE
            .find_iter(&prose)
            .map(|span| normalize_numeral_token(&prose[span.range()])),
    );
    numbers.extend(
        RUSSIAN_NUMERAL_RE
            .find_iter(&prose)
            .filter(|span| NONQUANTITY_ONE_RE.match_at(&prose, span.start).is_none())
            .map(|span| normalize_russian_integer(&prose[span.range()])),
    );
    numbers.extend(
        RUSSIAN_DURATION_RE
            .captures_iter(&prose)
            .map(|caps| normalize_russian_duration(&caps)),
    );
    let contextual = contextual_one_offsets(&prose);
    for span in ENGLISH_WORD_NUMERAL_RE.find_iter(&prose) {
        numbers.extend(
            english_number_pieces(&prose, span)
                .into_iter()
                .filter(|(start, _, _)| !is_nonquantity_one(&prose, *start, &contextual))
                .map(|(_, _, value)| value),
        );
    }
    numbers.sort();
    numbers
}

/// The unique normalized numerals (`_extract_normalized_numerals` default view).
pub fn extract_normalized_numerals(text: &str) -> std::collections::BTreeSet<String> {
    extract_numeral_occurrences(text).into_iter().collect()
}

// ── Numeric bounds ──

const NUMERIC_BOUND_PREFIXES: [(&str, &str); 6] = [
    (
        "lower_inclusive",
        r"at\s+least|(?:no|not)\s+(?:fewer|less)\s+than|не\s+(?:менее|меньше)|(?:как\s+)?минимум",
    ),
    (
        "upper_inclusive",
        r"at\s+most|(?:no|not)\s+(?:more|greater)\s+than|не\s+(?:более|больше)|(?:как\s+)?максимум",
    ),
    (
        "lower_exclusive",
        r"more\s+than|greater\s+than|over|более|больше|свыше",
    ),
    (
        "upper_exclusive",
        r"less\s+than|fewer\s+than|under|below|менее|меньше",
    ),
    (
        "approximate",
        r"about|around|approximately|roughly|около|примерно|приблизительно",
    ),
    ("estimate", r"estimated(?:\s+at)?|(?:an?\s+)?estimate\s+of"),
];

pub(crate) static NUMERIC_BOUND_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    let prefixes: Vec<String> = NUMERIC_BOUND_PREFIXES
        .iter()
        .map(|(name, pattern)| format!("(?P<{name}>{pattern})"))
        .collect();
    PyRegex::new(
        &format!(
            r"\b(?:{})\s+(?P<quantity>{}|{INDEFINITE_DURATION_PATTERN}|{}|{}|{NUMERAL_TOKEN}|{})(?!\w)",
            prefixes.join("|"),
            *MEASUREMENT_PATTERN,
            *RUSSIAN_DURATION,
            *RUSSIAN_NUMERAL,
            *ENGLISH_WORD_NUMERAL
        ),
        Flags::I,
    )
});

/// A `(kind, normalized value)` bound.
pub type NumericBound = (String, String);

/// Bind selected comparison, approximation and estimate prefixes to their values.
pub fn extract_numeric_bounds(text: &str) -> std::collections::BTreeSet<NumericBound> {
    let content = without_placeholders(text);
    let mut bounds = std::collections::BTreeSet::new();
    for caps in NUMERIC_BOUND_RE.captures_iter(&content) {
        let Some(kind) = NUMERIC_BOUND_PREFIXES
            .iter()
            .map(|(name, _)| *name)
            .find(|name| caps.name(name).is_some())
        else {
            continue;
        };
        let Some(quantity_span) = group_span(&caps, "quantity") else {
            continue;
        };
        if kind == "approximate"
            && NONQUANTITY_ONE_RE
                .match_at(&content, quantity_span.start)
                .is_some()
        {
            continue;
        }
        if kind == "lower_exclusive"
            && group(&caps, kind).is_some_and(|cue| unicode::casefold(cue) == "over")
            && RECIPROCAL_ONE_RE
                .match_at(&content, quantity_span.start)
                .is_some()
        {
            continue;
        }
        let original = &content[quantity_span.range()];
        let quantity = unicode::lower(original);
        if let Some(measurement) = MEASUREMENT_RE.full_captures(original) {
            let (value, unit) = measurement_signature(&measurement);
            bounds.insert((kind.to_owned(), format!("measurement:{unit}:{value}")));
            continue;
        }
        if INDEFINITE_DURATION_RE.fullmatch(&quantity) {
            let (value, unit) = indefinite_duration_signature(&quantity);
            bounds.insert((kind.to_owned(), format!("measurement:{unit}:{value}")));
            continue;
        }
        if let Some(duration) = RUSSIAN_DURATION_RE.full_captures(&quantity) {
            let (value, unit) = duration_measurement_signature(&duration);
            bounds.insert((kind.to_owned(), format!("measurement:{unit}:{value}")));
            continue;
        }
        let normalized = if RUSSIAN_NUMERAL_RE.fullmatch(&quantity) {
            normalize_russian_integer(&quantity)
        } else if ENGLISH_WORD_NUMERAL_RE.fullmatch(&quantity) {
            normalize_english_integer(&quantity)
        } else {
            normalize_numeral_token(&quantity)
        };
        bounds.insert((kind.to_owned(), normalized));
    }
    bounds
}

/// Selected adjacent English change phrases with explicit percent roles.
pub fn extract_explicit_percent_change_roles(
    text: &str,
) -> IndexMap<String, std::collections::BTreeSet<&'static str>> {
    let mut roles: IndexMap<String, std::collections::BTreeSet<&'static str>> = IndexMap::new();
    for caps in PERCENT_CHANGE_RE.captures_iter(text) {
        let (value, _) = measurement_signature(&caps);
        let link = group(&caps, "noun_link")
            .or_else(|| group(&caps, "verb_link"))
            .map(unicode::lower)
            .unwrap_or_default();
        roles
            .entry(value)
            .or_default()
            .insert(if link == "to" { "endpoint" } else { "change" });
    }
    roles
}

pub fn preserves_explicit_percent_change_roles(source: &str, candidate: &str) -> bool {
    let source_roles = extract_explicit_percent_change_roles(source);
    let candidate_roles = extract_explicit_percent_change_roles(candidate);
    source_roles
        .iter()
        .all(|(value, left)| match candidate_roles.get(value) {
            Some(right) if left.len() == 1 && right.len() == 1 => left == right,
            _ => true,
        })
}

/// Pattern sources for parity checks against the Python reference.
pub(crate) fn pattern_sources() -> Vec<(&'static str, String)> {
    vec![
        ("_PLACEHOLDER_TOKEN_RE", PLACEHOLDER_TOKEN.to_owned()),
        ("_PLACEHOLDER_LIKE_RE", PLACEHOLDER_LIKE.to_owned()),
        ("_NUMERAL_TOKEN_RE", NUMERAL_TOKEN.to_owned()),
        ("_ENGLISH_INTEGER_RE", ENGLISH_INTEGER.clone()),
        ("_ENGLISH_WORD_NUMERAL_RE", ENGLISH_WORD_NUMERAL.clone()),
        ("_FREQUENCY_PATTERN", FREQUENCY_PATTERN.clone()),
        ("_FREQUENCY_REPEAT_RE", FREQUENCY_REPEAT_RE.source().to_owned()),
        (
            "_RUSSIAN_NUMBER_WORD_PATTERN",
            RUSSIAN_NUMBER_WORD_PATTERN.clone(),
        ),
        ("_RUSSIAN_NUMERAL_RE", RUSSIAN_NUMERAL.clone()),
        ("_RUSSIAN_DURATION_RE", RUSSIAN_DURATION.clone()),
        ("_MEASUREMENT_ATOM_PATTERN", MEASUREMENT_ATOM_PATTERN.clone()),
        ("_MEASUREMENT_AMOUNT_PATTERN", MEASUREMENT_AMOUNT_PATTERN.clone()),
        ("_MEASUREMENT_UNIT_PATTERN", MEASUREMENT_UNIT_PATTERN.clone()),
        ("_MEASUREMENT_PATTERN", MEASUREMENT_PATTERN.clone()),
        ("_MEASUREMENT_RE", MEASUREMENT_RE.source().to_owned()),
        ("_PERCENT_CHANGE_RE", PERCENT_CHANGE_RE.source().to_owned()),
        (
            "_INDEFINITE_DURATION_PATTERN",
            INDEFINITE_DURATION_PATTERN.to_owned(),
        ),
        ("_VAGUE_DURATION_PATTERN", VAGUE_DURATION_PATTERN.to_owned()),
        ("_WEEKEND_TIMING_PATTERN", WEEKEND_TIMING_PATTERN.to_owned()),
        ("_NUMERIC_BOUND_RE", NUMERIC_BOUND_RE.source().to_owned()),
        ("_NONQUANTITY_ONE_RE", NONQUANTITY_ONE_RE.source().to_owned()),
        ("_RECIPROCAL_ONE_RE", RECIPROCAL_ONE_RE.source().to_owned()),
        (
            "_GENERIC_KNOWLEDGE_SUBJECT_RE",
            GENERIC_KNOWLEDGE_SUBJECT_RE.source().to_owned(),
        ),
        (
            "_INDEFINITE_PARTICIPLE_ONE_RE",
            INDEFINITE_PARTICIPLE_ONE_RE.source().to_owned(),
        ),
    ]
}
