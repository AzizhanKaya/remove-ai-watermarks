//! Typed forms of the audio CLI strings (`AudioScheme`, `AudioMethod`, `AudioProfile`, vendors, devices).

use std::fmt;
use std::str::FromStr;

use raiw_core::constants::AUDIO_VENDORS;
use serde::{Serialize, Serializer};

use crate::decoders::Decoder;
use crate::error::AudioError;

macro_rules! string_enum {
    ($name:ident, $error:expr, { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = AudioError;
            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::ALL.iter().copied().find(|v| v.as_str() == value).ok_or_else(|| AudioError::invalid($error))
            }
        }
    };
}

string_enum!(Scheme, "Audio scheme must be audioseal, wavmark, perth, silentcipher, all, or auto", {
    AudioSeal => "audioseal",
    WavMark => "wavmark",
    Perth => "perth",
    SilentCipher => "silentcipher",
    All => "all",
    Auto => "auto",
});

string_enum!(Method, "Audio method must be auto, spectral, encodec, combined, dac, or snac", {
    Auto => "auto",
    Spectral => "spectral",
    Encodec => "encodec",
    Combined => "combined",
    Dac => "dac",
    Snac => "snac",
});

string_enum!(Profile, "Audio method must be spectral, encodec, combined, dac, or snac", {
    Spectral => "spectral",
    Encodec => "encodec",
    Combined => "combined",
    Dac => "dac",
    Snac => "snac",
});

string_enum!(Device, "Vocoder device must be auto, cpu, mps, or cuda", {
    Auto => "auto",
    Cpu => "cpu",
    Mps => "mps",
    Cuda => "cuda",
});

string_enum!(Vendor, "Audio vendor must be one of: auto, google, elevenlabs, suno, udio, openai, cartesia, meta, resemble, microsoft", {
    Google => "google",
    ElevenLabs => "elevenlabs",
    Suno => "suno",
    Udio => "udio",
    OpenAi => "openai",
    Cartesia => "cartesia",
    Meta => "meta",
    Resemble => "resemble",
    Microsoft => "microsoft",
});

// A proprietary waveform vendor recognized from pre-strip metadata.
string_enum!(MetadataVendor, "unknown metadata vendor", {
    ElevenLabs => "elevenlabs",
    GoogleSynthId => "google_synthid",
    Cartesia => "cartesia",
    OpenAi => "openai",
    Suno => "suno",
});

impl Scheme {
    /// The single named decoder of a non-aggregate scheme.
    pub const fn decoder(self) -> Option<Decoder> {
        match self {
            Self::AudioSeal => Some(Decoder::AudioSeal),
            Self::WavMark => Some(Decoder::WavMark),
            Self::Perth => Some(Decoder::Perth),
            Self::SilentCipher => Some(Decoder::SilentCipher),
            Self::All | Self::Auto => None,
        }
    }

    /// Decoders requested by this scheme, in report order.
    pub fn requested(self) -> Vec<Decoder> {
        match self {
            Self::All => Decoder::NAMED.to_vec(),
            Self::Auto => Decoder::AUTO.to_vec(),
            other => other.decoder().into_iter().collect(),
        }
    }

    /// `_AUTO_PROFILES[scheme]`: the fixed `--method auto` sequence per scheme.
    pub const fn auto_profiles(self) -> &'static [Profile] {
        match self {
            Self::AudioSeal | Self::Perth | Self::Auto => &[Profile::Spectral, Profile::Combined],
            Self::WavMark | Self::SilentCipher => &[Profile::Encodec, Profile::Combined],
            Self::All => &[Profile::Combined],
        }
    }

    /// The CLI's `_audio_check_label`.
    pub const fn check_label(self) -> &'static str {
        match self {
            Self::AudioSeal => "open AudioSeal",
            Self::WavMark => "public WavMark",
            Self::Perth => "public Perth v1",
            Self::SilentCipher => "public SilentCipher-44k",
            Self::All => "AudioSeal, WavMark, Perth v1, and SilentCipher-44k",
            Self::Auto => "installed compatible public audio watermark",
        }
    }
}

impl Method {
    pub const fn profile(self) -> Option<Profile> {
        match self {
            Self::Auto => None,
            Self::Spectral => Some(Profile::Spectral),
            Self::Encodec => Some(Profile::Encodec),
            Self::Combined => Some(Profile::Combined),
            Self::Dac => Some(Profile::Dac),
            Self::Snac => Some(Profile::Snac),
        }
    }

    /// Methods whose candidate set includes EnCodec and so validates its bandwidth.
    pub const fn uses_bandwidth(self) -> bool {
        matches!(self, Self::Auto | Self::Encodec | Self::Combined)
    }
}

impl Profile {
    /// EnCodec runs (alone or after the spectral filter).
    pub const fn is_neural(self) -> bool {
        matches!(self, Self::Encodec | Self::Combined)
    }

    /// Full-track raw codecs on the 44.1 kHz rational grid.
    pub const fn is_raw_codec(self) -> bool {
        matches!(self, Self::Dac | Self::Snac)
    }

    pub const fn uses_spectral(self) -> bool {
        matches!(self, Self::Spectral | Self::Combined)
    }
}

impl From<Profile> for Method {
    fn from(profile: Profile) -> Self {
        match profile {
            Profile::Spectral => Self::Spectral,
            Profile::Encodec => Self::Encodec,
            Profile::Combined => Self::Combined,
            Profile::Dac => Self::Dac,
            Profile::Snac => Self::Snac,
        }
    }
}

/// `normalize_audio_vendor`: `None`/`"auto"` mean no explicit cohort.
pub fn parse_vendor(value: Option<&str>) -> Result<Option<Vendor>, AudioError> {
    match value {
        None | Some("auto") => Ok(None),
        Some(text) => Vendor::ALL
            .iter()
            .copied()
            .find(|v| v.as_str() == text)
            .map(Some)
            .ok_or_else(|| {
                AudioError::invalid(format!(
                    "Audio vendor must be one of: {}",
                    AUDIO_VENDORS.join(", ")
                ))
            }),
    }
}

/// EnCodec bandwidths in kbps (`AUDIO_BANDWIDTHS_KBPS`).
pub const BANDWIDTHS_KBPS: [f64; 3] = [6.0, 12.0, 24.0];

pub fn valid_bandwidth(kbps: f64) -> bool {
    BANDWIDTHS_KBPS.contains(&kbps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_strings_roundtrip() {
        for scheme in Scheme::ALL {
            assert_eq!(scheme.as_str().parse::<Scheme>().unwrap(), *scheme);
        }
        assert!("nope".parse::<Method>().is_err());
        assert_eq!(parse_vendor(Some("auto")).unwrap(), None);
        assert_eq!(parse_vendor(Some("meta")).unwrap(), Some(Vendor::Meta));
        assert_eq!(
            parse_vendor(Some("x")).unwrap_err().to_string(),
            "Audio vendor must be one of: auto, google, elevenlabs, suno, udio, openai, cartesia, meta, resemble, microsoft"
        );
    }
}
