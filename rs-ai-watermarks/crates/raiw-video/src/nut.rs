//! A minimal NUT muxer for rawvideo with explicit PTS (`_TimestampedNutWriter`).
//!
//! Python muxes through PyAV; this writes the same stream shape directly: one rawvideo stream in
//! the source time base, every frame a keyframe preceded by a syncpoint, full PTS coded per frame.
//! Packets carry the NUT CRC (polynomial 0x04C11DB7, MSB-first, zero init, stored big-endian).

use std::io::Write;

use crate::error::{Result, VideoError};
use crate::probe::Rational;

const ID_STRING: &[u8] = b"nut/multimedia container\0";
const fn startcode(low: u64, tag: u8) -> u64 {
    low + ((((b'N' as u64) << 8) | tag as u64) << 48)
}
const MAIN_STARTCODE: u64 = startcode(0x7A56_1F5F_04AD, b'M');
const STREAM_STARTCODE: u64 = startcode(0x1140_5BF2_F9DB, b'S');
const SYNCPOINT_STARTCODE: u64 = startcode(0xE4AD_EECA_4569, b'K');

const FLAG_KEY: u64 = 1;
const FLAG_CODED_PTS: u64 = 8;
const FLAG_SIZE_MSB: u64 = 32;
const FLAG_CHECKSUM: u64 = 64;
const FLAG_CODED: u64 = 4096;
const MSB_PTS_SHIFT: u64 = 7;
const MAX_DISTANCE: u64 = 32768;

fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut c = (i as u32) << 24;
            for _ in 0..8 {
                c = if c & 0x8000_0000 != 0 {
                    (c << 1) ^ 0x04C1_1DB7
                } else {
                    c << 1
                };
            }
            *entry = c;
        }
        table
    });
    data.iter().fold(0u32, |crc, &byte| {
        (crc << 8) ^ table[((crc >> 24) as u8 ^ byte) as usize]
    })
}

fn put_v(out: &mut Vec<u8>, value: u64) {
    let mut groups = 1;
    while groups < 10 && value >> (7 * groups) != 0 {
        groups += 1;
    }
    for index in (0..groups).rev() {
        let byte = ((value >> (7 * index)) & 0x7F) as u8;
        out.push(if index == 0 { byte } else { byte | 0x80 });
    }
}

fn put_s(out: &mut Vec<u8>, value: i64) {
    let coded = if value <= 0 {
        (-2 * value) as u64
    } else {
        (2 * value - 1) as u64
    };
    put_v(out, coded);
}

fn packet(code: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 24);
    out.extend_from_slice(&code.to_be_bytes());
    let forward = payload.len() as u64 + 4;
    let header_start = out.len() - 8;
    put_v(&mut out, forward);
    if forward > 4096 {
        let crc = crc32(&out[header_start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32(payload).to_be_bytes());
    out
}

/// The rawvideo layouts the NUT stream can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NutPixelFormat {
    Bgr24,
    Bgr48Le,
}

impl NutPixelFormat {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bgr24" => Some(Self::Bgr24),
            "bgr48le" => Some(Self::Bgr48Le),
            _ => None,
        }
    }

    const fn fourcc(self) -> [u8; 4] {
        match self {
            Self::Bgr24 => [b'B', b'G', b'R', 24],
            Self::Bgr48Le => [b'B', b'G', b'R', 48],
        }
    }

    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Bgr24 => 3,
            Self::Bgr48Le => 6,
        }
    }
}

/// Streams BGR frames with explicit PTS into an ffmpeg `-f nut -i pipe:0` input.
#[derive(Debug)]
pub struct TimestampedNutWriter<W: Write> {
    out: W,
    time_base: Rational,
    start_pts: i64,
    origin: Option<f64>,
    frame_size: usize,
    last_pts: Option<i64>,
}

impl<W: Write> TimestampedNutWriter<W> {
    pub fn new(
        mut out: W,
        width: usize,
        height: usize,
        time_base: Rational,
        start_pts: i64,
        pix_fmt: NutPixelFormat,
    ) -> Result<Self> {
        if width == 0 || height == 0 || !time_base.is_positive() {
            return Err(VideoError::value(
                "NUT stream needs positive geometry and time base",
            ));
        }
        let mut main = Vec::new();
        put_v(&mut main, 3);
        put_v(&mut main, 1);
        put_v(&mut main, MAX_DISTANCE);
        put_v(&mut main, 1);
        put_v(&mut main, time_base.num as u64);
        put_v(&mut main, time_base.den as u64);
        put_v(&mut main, FLAG_CODED);
        put_v(&mut main, 6);
        put_s(&mut main, 0);
        put_v(&mut main, 1);
        put_v(&mut main, 0);
        put_v(&mut main, 0);
        put_v(&mut main, 0);
        put_v(&mut main, 255);
        put_v(&mut main, 0);

        let mut stream = Vec::new();
        put_v(&mut stream, 0);
        put_v(&mut stream, 0);
        put_v(&mut stream, 4);
        stream.extend_from_slice(&pix_fmt.fourcc());
        put_v(&mut stream, 0);
        put_v(&mut stream, MSB_PTS_SHIFT);
        put_v(&mut stream, time_base.den as u64);
        put_v(&mut stream, 0);
        put_v(&mut stream, 0);
        put_v(&mut stream, 0);
        put_v(&mut stream, width as u64);
        put_v(&mut stream, height as u64);
        put_v(&mut stream, 0);
        put_v(&mut stream, 0);
        put_v(&mut stream, 0);

        out.write_all(ID_STRING)?;
        out.write_all(&packet(MAIN_STARTCODE, &main))?;
        out.write_all(&packet(STREAM_STARTCODE, &stream))?;
        Ok(Self {
            out,
            time_base,
            start_pts,
            origin: None,
            frame_size: width * height * pix_fmt.bytes_per_pixel(),
            last_pts: None,
        })
    }

    /// Mux one frame at its source-timeline timestamp (seconds).
    pub fn write(&mut self, frame: &[u8], timestamp: f64) -> Result<()> {
        if frame.len() != self.frame_size {
            return Err(VideoError::value(
                "NUT frame size does not match the stream geometry",
            ));
        }
        let origin = *self.origin.get_or_insert(timestamp);
        let ticks = ((timestamp - origin) / self.time_base.to_f64()).round_ties_even() as i64;
        let pts = self.start_pts + ticks;
        if pts < 0 || self.last_pts.is_some_and(|last| pts <= last) {
            return Err(VideoError::runtime(format!(
                "Timestamped frame PTS {pts} is negative or not strictly increasing"
            )));
        }
        self.last_pts = Some(pts);
        let mut sync = Vec::new();
        put_v(&mut sync, pts as u64);
        put_v(&mut sync, 0);
        self.out.write_all(&packet(SYNCPOINT_STARTCODE, &sync))?;

        let mut header = vec![0u8];
        put_v(
            &mut header,
            FLAG_KEY | FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM,
        );
        put_v(&mut header, pts as u64 + (1 << MSB_PTS_SHIFT));
        put_v(&mut header, frame.len() as u64);
        let crc = crc32(&header);
        header.extend_from_slice(&crc.to_be_bytes());
        self.out.write_all(&header)?;
        self.out.write_all(frame)?;
        Ok(())
    }

    /// Flush without closing the underlying pipe.
    pub fn finish(mut self) -> Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_match_the_reference_encoding() {
        let mut out = Vec::new();
        put_v(&mut out, 32767);
        assert_eq!(out, [0x81, 0xFF, 0x7F]);
        out.clear();
        put_v(&mut out, 51200);
        assert_eq!(out, [0x83, 0x90, 0x00]);
        out.clear();
        put_s(&mut out, 0);
        put_s(&mut out, 1);
        put_s(&mut out, -1);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn crc_matches_ffmpeg_packets() {
        // The stream-header payload of an ffmpeg-written 64x32 bgr24 NUT file.
        let payload = [
            0, 0, 4, 0x42, 0x47, 0x52, 0x18, 0, 0x0e, 0x83, 0x90, 0, 0, 0, 0, 0x40, 0x20, 1, 1, 0,
        ];
        assert_eq!(crc32(&payload), 0x0a4c_ca9a);
    }
}
