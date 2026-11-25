//! Registered TC260 producer identities, mark routing, and pinned signing keys
//! (port of `_internal/tc260_producers.py`).

/// One producer's measured codes and independently established identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tc260Producer {
    pub key: &'static str,
    pub codes: &'static [&'static str],
    pub vendor: &'static str,
    pub image_mark: Option<&'static str>,
    pub video_mark: Option<&'static str>,
    pub video_platform: Option<&'static str>,
    pub signer: Option<&'static str>,
    /// Big-endian x coordinate of the pinned SM2 signing key (even y).
    pub signing_key_x: Option<[u8; 32]>,
    pub image_platform: Option<&'static str>,
}

impl Tc260Producer {
    const fn new(key: &'static str, codes: &'static [&'static str], vendor: &'static str) -> Self {
        Self {
            key,
            codes,
            vendor,
            image_mark: None,
            video_mark: None,
            video_platform: None,
            signer: None,
            signing_key_x: None,
            image_platform: None,
        }
    }

    const fn image_mark(mut self, mark: &'static str) -> Self {
        self.image_mark = Some(mark);
        self
    }

    const fn video_mark(mut self, mark: &'static str) -> Self {
        self.video_mark = Some(mark);
        self
    }

    const fn video_platform(mut self, platform: &'static str) -> Self {
        self.video_platform = Some(platform);
        self
    }

    const fn image_platform(mut self, platform: &'static str) -> Self {
        self.image_platform = Some(platform);
        self
    }

    const fn platforms(self, platform: &'static str) -> Self {
        self.video_platform(platform).image_platform(platform)
    }

    const fn signed_by(mut self, signer: &'static str, key_x: [u8; 32]) -> Self {
        self.signer = Some(signer);
        self.signing_key_x = Some(key_x);
        self
    }
}

const MINIMAX_KEY_X: [u8; 32] = [
    0xA0, 0xB3, 0xB0, 0xB6, 0xA0, 0xC9, 0xB0, 0xC8, 0x9C, 0xAB, 0x32, 0x83, 0x42, 0xAF, 0x4E, 0x82, 0x21, 0xEC,
    0x5B, 0x40, 0x79, 0x9C, 0xBE, 0x83, 0x5A, 0xB4, 0x25, 0x1F, 0x7B, 0x47, 0xE4, 0xFD,
];

pub const TC260_PRODUCERS: &[Tc260Producer] = &[
    // YOYO JPEG exports and Honor AI video clips
    Tc260Producer::new("honor", &["91440300MA5G49LC9K", "HZ_XYKJ"], "Honor").platforms("Honor"),
    Tc260Producer::new("doubao", &["91110102MACQD9K640", "doubao"], "ByteDance")
        .image_mark("doubao")
        .video_mark("doubao")
        .video_platform("ByteDance Doubao"),
    Tc260Producer::new("jimeng", &["9144030008867405X2"], "ByteDance")
        .image_mark("jimeng")
        .video_mark("dreamina")
        .video_platform("ByteDance Dreamina"),
    Tc260Producer::new("volcengine", &["91110108MA01R70K8D"], "ByteDance").platforms("ByteDance Volcano Engine"),
    Tc260Producer::new("qwen", &["91440101MA9Y9T4H7A"], "Alibaba")
        .image_mark("qwen")
        .video_mark("qwen")
        .video_platform("Alibaba Cloud Qwen"),
    // Tongyi Yunqi signs both Wan and HappyHorse exports
    Tc260Producer::new("wan", &["91330106MA2CFLDG4R"], "Alibaba")
        .image_mark("wan")
        .video_mark("wan")
        .video_platform("Alibaba Tongyi (Wan, HappyHorse)"),
    Tc260Producer::new("kling", &["91110108335469089C", "91110108MAKBA45N9M", "kling"], "Kuaishou")
        .image_mark("kling")
        .video_mark("kling")
        .platforms("Kuaishou Kling AI"),
    Tc260Producer::new("yuanbao", &["91440300708461136T", "9144030071526726XG", "91440101327598294H"], "Tencent")
        .image_mark("yuanbao")
        .video_mark("yuanbao")
        .platforms("Tencent Yuanbao"),
    Tc260Producer::new("workbuddy", &["911101085636549482"], "Tencent")
        .image_mark("workbuddy")
        .platforms("Tencent WorkBuddy"),
    Tc260Producer::new("runninghub", &["91340100MAEB4N8H76", "RunningHub"], "RunningHub")
        .image_mark("runninghub")
        .video_mark("runninghub"),
    Tc260Producer::new("baidu", &["91110000802100433B"], "Baidu")
        .image_mark("baidu")
        .video_mark("baidu")
        .video_platform("Baidu"),
    Tc260Producer::new("liblib", &["91110105MACJ6K1C8A"], "LiblibAI").image_mark("liblib"),
    Tc260Producer::new("meitu", &["913502007378955153"], "Meitu").image_platform("Meitu"),
    Tc260Producer::new("vivo", &["91441900557262083U"], "vivo").image_platform("vivo"),
    Tc260Producer::new("xiaomi", &["91110108596084056A"], "Xiaomi").image_platform("Xiaomi"),
    // Apple Technology Development (Shanghai) Co., Ltd.
    Tc260Producer::new("apple", &["91310115MAEA70M85H"], "Apple").platforms("Apple Intelligence"),
    // Samsung (China) Investment Co., Ltd.
    Tc260Producer::new("samsung_cn", &["91110000625911349W"], "Samsung").platforms("Samsung"),
    // Huawei Software Technologies Co., Ltd.
    Tc260Producer::new("huawei", &["913201147770231720"], "Huawei").platforms("Huawei"),
    // OnePlus Technology (Shenzhen) Co., Ltd.
    Tc260Producer::new("oneplus", &["91440300080768342L", "ONEPLUS"], "OnePlus").platforms("OnePlus"),
    Tc260Producer::new("agnes", &["91320191MAKHJ19D9J"], "Agnes AI").platforms("Agnes AI"),
    // ShengShu, verified against the Beijing municipal list of 2025-03-03
    Tc260Producer::new("vidu", &["91110108MACC4D63XF"], "ShengShu").video_mark("vidu").video_platform("ShengShu Vidu"),
    // Even-y SM2 point for the x coordinate carried by measured MiniMax exports
    Tc260Producer::new("minimax", &["MiniMax"], "MiniMax")
        .video_mark("hailuo")
        .video_platform("MiniMax Hailuo AI")
        .signed_by("MiniMax", MINIMAX_KEY_X),
];

/// Extract the USCC from `001` + type digit + USCC + product suffix.
pub fn uscc_of(code: &str) -> &str {
    if code.len() >= 22 && code.starts_with("001") && code.is_char_boundary(4) && code.is_char_boundary(22) {
        &code[4..22]
    } else {
        code
    }
}

fn casefold(text: &str) -> String {
    text.to_lowercase()
}

/// Match a whole producer identity, USCC-normalized and casefolded.
pub fn tc260_producer_in<'a>(producer: &str, codes: impl IntoIterator<Item = &'a str>) -> bool {
    let code = casefold(uscc_of(producer.trim()));
    !code.is_empty() && codes.into_iter().any(|candidate| code == casefold(candidate))
}

/// Resolve a producer field; partial names never establish an identity.
pub fn producer_for_code(code: &str) -> Option<&'static Tc260Producer> {
    let wanted = casefold(uscc_of(code.trim()));
    TC260_PRODUCERS.iter().find(|row| row.codes.iter().any(|c| casefold(c) == wanted))
}

/// Resolve the identity returned by successful pinned-key verification.
pub fn producer_for_signer(signer: &str) -> Option<&'static Tc260Producer> {
    TC260_PRODUCERS.iter().find(|row| row.signer == Some(signer))
}

/// Measured producer identities for an image or video mark, if registered.
pub fn producer_codes_for_mark(mark: &str) -> &'static [&'static str] {
    TC260_PRODUCERS
        .iter()
        .find(|row| row.image_mark == Some(mark) || row.video_mark == Some(mark))
        .map_or(&[], |row| row.codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_resolve_through_uscc_and_casefold() {
        assert_eq!(producer_for_code(" doubao ").map(|p| p.key), Some("doubao"));
        assert_eq!(producer_for_code("001991110102MACQD9K640XYZ").map(|p| p.key), Some("doubao"));
        assert_eq!(producer_for_code("douba"), None);
        assert_eq!(uscc_of("00199144030008867405X2abc"), "9144030008867405X2");
        assert_eq!(uscc_of("short"), "short");
    }

    #[test]
    fn marks_signers_and_membership() {
        assert_eq!(producer_codes_for_mark("hailuo"), &["MiniMax"]);
        assert_eq!(producer_codes_for_mark("dreamina"), &["9144030008867405X2"]);
        assert!(producer_codes_for_mark("unknown").is_empty());
        assert_eq!(producer_for_signer("MiniMax").map(|p| p.key), Some("minimax"));
        assert!(tc260_producer_in("KLING", ["kling"]));
        assert!(!tc260_producer_in("", ["kling"]));
    }
}
