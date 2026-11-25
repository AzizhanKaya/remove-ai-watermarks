//! Single-decode multi-decoder scanning (`_scan_audio_multi`, `_WindowSegmentAccumulator`).
//!
//! One ffmpeg pass feeds shared 16 kHz windows to AudioSeal, WavMark and Perth, native-clock
//! AudioSeal for calibrated 24/48 kHz sources, the OpenVoice prefix, and 3-second SilentCipher windows.

use std::collections::HashMap;
use std::path::Path;

use crate::decoders::{self, Decoder, DecoderError, DecoderSet, ModelReading};
use crate::dsp::resample_rates;
use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::io::{AudioReader, AudioStreamInfo, FfmpegTools};
use crate::report::{AudioDetection, AudioWindowReading, Channel, CheckOutcome, channels_of, status};

pub const MAX_WHOLE_SECONDS: usize = 30;
pub const MAX_WHOLE_VALUES: usize = 16_000_000;
const SHORT_WINDOW: &str = "Audio is shorter than one complete model window";

/// What the accumulator asks its owner to decode.
#[derive(Debug)]
pub enum Event {
    Window {
        block: Frames,
        start: usize,
        end: usize,
    },
    /// `block` is `None` when the single whole segment equals the only window (reuse its reading).
    Segment {
        block: Option<Frames>,
        start: usize,
        end: usize,
    },
}

/// Tail-padded windows plus bounded overlapping whole-clip segments for one sample clock.
#[derive(Debug)]
pub struct WindowSegmentAccumulator {
    win_frames: usize,
    max_seg_frames: usize,
    prev: Frames,
    count: usize,
    seg: Frames,
    seg_start: usize,
    win_count: usize,
    single_bounds: Option<(usize, usize)>,
    seg_flushes: usize,
}

impl WindowSegmentAccumulator {
    pub fn new(channels: usize, win_frames: usize, max_seg_frames: usize) -> Self {
        Self {
            win_frames,
            max_seg_frames,
            prev: Frames::empty(channels),
            count: 0,
            seg: Frames::empty(channels),
            seg_start: 0,
            win_count: 0,
            single_bounds: None,
            seg_flushes: 0,
        }
    }

    /// The scanner's segment bound: 30 s, or the whole-clip value budget per channel.
    pub fn for_clock(channels: usize, sample_rate: u32, win_frames: usize) -> Self {
        let max_seg = (MAX_WHOLE_SECONDS * sample_rate as usize)
            .min(win_frames.max(MAX_WHOLE_VALUES / channels.max(1)));
        Self::new(channels, win_frames, max_seg)
    }

    pub fn feed(&mut self, frames: &Frames, events: &mut Vec<Event>) {
        let mut window = frames.clone();
        let mut start = self.count;
        if frames.frames() < self.win_frames && !self.prev.is_empty() {
            let mut padded = self.prev.tail(self.win_frames - frames.frames());
            padded.append(frames);
            start = self.count + frames.frames() - padded.frames();
            window = padded;
        }
        let end = self.count + frames.frames();
        if self.win_count == 0 {
            self.single_bounds = Some((start, end));
        }
        self.win_count += 1;
        events.push(Event::Window {
            block: window,
            start,
            end,
        });
        self.prev = frames.clone();
        self.count = end;

        let mut pending = frames.clone();
        while !pending.is_empty() {
            if self.seg.frames() >= self.max_seg_frames {
                let joined = std::mem::replace(&mut self.seg, Frames::empty(frames.channels()));
                self.seg_flushes += 1;
                let joined_len = joined.frames();
                let keep = self
                    .win_frames
                    .min(self.max_seg_frames.saturating_sub(1))
                    .min(joined_len);
                let overlap = joined.tail(keep);
                events.push(Event::Segment {
                    block: Some(joined),
                    start: self.seg_start,
                    end: self.seg_start + joined_len,
                });
                self.seg_start = self.seg_start + joined_len - overlap.frames();
                self.seg = overlap;
            }
            let space = self.max_seg_frames - self.seg.frames();
            let take = pending.slice(0..space);
            self.seg.append(&take);
            pending = pending.slice(take.frames()..pending.frames());
        }
    }

    pub fn finish(&mut self, events: &mut Vec<Event>) {
        if self.seg.is_empty() {
            return;
        }
        let end = self.seg_start + self.seg.frames();
        let reuse =
            self.seg_flushes == 0 && self.win_count == 1 && self.single_bounds == Some((self.seg_start, end));
        self.seg_flushes += 1;
        let block = std::mem::replace(&mut self.seg, Frames::empty(self.prev.channels()));
        events.push(Event::Segment {
            block: (!reuse).then_some(block),
            start: self.seg_start,
            end,
        });
    }
}

/// `_model_samples`: mono audio on AudioSeal's 16 kHz clock.
pub fn model_samples(mono: &[f32], sample_rate: u32) -> Vec<f32> {
    if sample_rate == decoders::audioseal::SAMPLE_RATE {
        return mono.to_vec();
    }
    resample_rates(mono, sample_rate, decoders::audioseal::SAMPLE_RATE)
}

fn outcome(result: std::result::Result<ModelReading, DecoderError>) -> Result<CheckOutcome> {
    match result {
        Ok(reading) => Ok(CheckOutcome::read(reading)),
        Err(DecoderError::Invalid(message)) => Err(AudioError::Invalid(message)),
        Err(DecoderError::Failed(message) | DecoderError::ModelCheck(message)) => {
            Ok(CheckOutcome::failed(message))
        }
    }
}

/// `_read_prepared_model` for the one-second decoders on prepared (16 kHz or native) samples.
fn read_prepared(decoders: &DecoderSet, scheme: Decoder, prepared: &[f32]) -> Result<CheckOutcome> {
    match scheme {
        Decoder::AudioSeal => {
            let net = decoders.audioseal.as_deref().ok_or_else(|| missing(scheme))?;
            outcome(decoders::audioseal::read(net, prepared).map(ModelReading::AudioSeal))
        }
        Decoder::WavMark | Decoder::Perth if prepared.len() < 16_000 => {
            Ok(CheckOutcome::inapplicable(SHORT_WINDOW))
        }
        Decoder::WavMark => {
            let net = decoders.wavmark.as_deref().ok_or_else(|| missing(scheme))?;
            outcome(decoders::wavmark::read(net, prepared).map(ModelReading::WavMark))
        }
        Decoder::Perth => {
            let net = decoders.perth.as_deref().ok_or_else(|| missing(scheme))?;
            outcome(decoders::perth::read(net, prepared).map(ModelReading::Perth))
        }
        Decoder::SilentCipher => Err(missing(scheme)),
    }
}

/// `_read_model(..., "silentcipher")`: every failure, including bad input, stays unverified.
fn read_silentcipher(decoders: &DecoderSet, mono: &[f32], sample_rate: u32) -> Result<CheckOutcome> {
    let net = decoders
        .silentcipher
        .as_deref()
        .ok_or_else(|| missing(Decoder::SilentCipher))?;
    Ok(match decoders::silentcipher::read(net, mono, sample_rate) {
        Ok(reading) => CheckOutcome::read(ModelReading::SilentCipher(reading)),
        Err(err) => CheckOutcome::failed(err.to_string()),
    })
}

fn missing(scheme: Decoder) -> AudioError {
    AudioError::unavailable(format!("{scheme} decoding is unavailable"))
}

fn mono_of(block: &Frames, channel: Channel) -> Vec<f32> {
    match channel {
        Channel::Mix => block.downmix(),
        Channel::Index(c) => block.channel(c),
    }
}

/// `(scheme, native AudioSeal clock)`.
type ClockKey = (Decoder, bool);
type PerChannel = HashMap<Channel, Vec<AudioWindowReading>>;

struct Lane {
    windows: PerChannel,
    segments: PerChannel,
}

impl Lane {
    fn new(channels: &[Channel]) -> Self {
        Self {
            windows: channels.iter().map(|&c| (c, Vec::new())).collect(),
            segments: channels.iter().map(|&c| (c, Vec::new())).collect(),
        }
    }
}

struct Scanner<'a> {
    decoders: &'a DecoderSet,
    info: &'a AudioStreamInfo,
    channels: Vec<Channel>,
    one_second: Vec<Decoder>,
    native_audioseal: bool,
    lanes: HashMap<ClockKey, Lane>,
    silentcipher: Lane,
}

impl Scanner<'_> {
    fn clock_keys(&self) -> Vec<ClockKey> {
        let mut keys: Vec<ClockKey> = self.one_second.iter().map(|&s| (s, false)).collect();
        if self.native_audioseal {
            keys.push((Decoder::AudioSeal, true));
        }
        keys
    }

    fn eval_one_second(&mut self, block: &Frames, start: usize, end: usize, segment: bool) -> Result<()> {
        for channel in self.channels.clone() {
            let mono = mono_of(block, channel);
            let prepared = model_samples(&mono, self.info.sample_rate);
            for scheme in self.one_second.clone() {
                let check = read_prepared(self.decoders, scheme, &prepared)?;
                self.push(
                    (scheme, false),
                    channel,
                    AudioWindowReading::new(start, end, check),
                    segment,
                );
            }
            if self.native_audioseal {
                let check = read_prepared(self.decoders, Decoder::AudioSeal, &mono)?;
                self.push(
                    (Decoder::AudioSeal, true),
                    channel,
                    AudioWindowReading::new(start, end, check),
                    segment,
                );
            }
        }
        Ok(())
    }

    fn push(&mut self, key: ClockKey, channel: Channel, reading: AudioWindowReading, segment: bool) {
        if let Some(lane) = self.lanes.get_mut(&key) {
            let target = if segment {
                &mut lane.segments
            } else {
                &mut lane.windows
            };
            target.entry(channel).or_default().push(reading);
        }
    }

    fn on_one_second(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Window { block, start, end } => self.eval_one_second(&block, start, end, false),
            Event::Segment {
                block: Some(block),
                start,
                end,
            } => self.eval_one_second(&block, start, end, true),
            Event::Segment { block: None, .. } => {
                for key in self.clock_keys() {
                    if let Some(lane) = self.lanes.get_mut(&key) {
                        for channel in &self.channels {
                            if let Some(first) = lane.windows.get(channel).and_then(|w| w.first()).cloned() {
                                lane.segments.entry(*channel).or_default().push(first);
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn on_silentcipher(&mut self, event: Event) -> Result<()> {
        let (block, start, end, segment) = match event {
            Event::Window { block, start, end } => (block, start, end, false),
            Event::Segment {
                block: Some(block),
                start,
                end,
            } => (block, start, end, true),
            Event::Segment { block: None, .. } => {
                for channel in &self.channels {
                    if let Some(first) = self
                        .silentcipher
                        .windows
                        .get(channel)
                        .and_then(|w| w.first())
                        .cloned()
                    {
                        self.silentcipher
                            .segments
                            .entry(*channel)
                            .or_default()
                            .push(first);
                    }
                }
                return Ok(());
            }
        };
        for channel in self.channels.clone() {
            let check = read_silentcipher(self.decoders, &mono_of(&block, channel), self.info.sample_rate)?;
            let target = if segment {
                &mut self.silentcipher.segments
            } else {
                &mut self.silentcipher.windows
            };
            target
                .entry(channel)
                .or_default()
                .push(AudioWindowReading::new(start, end, check));
        }
        Ok(())
    }
}

/// `_finalize_scheme_detections`.
fn finalize(
    scheme: Decoder,
    native_audioseal: bool,
    info: &AudioStreamInfo,
    channels: &[Channel],
    lane: &Lane,
) -> (Vec<AudioDetection>, Vec<String>) {
    let mut multi_segment = false;
    let detections: Vec<AudioDetection> = channels
        .iter()
        .map(|channel| {
            let segments = lane.segments.get(channel).cloned().unwrap_or_default();
            let windows = lane.windows.get(channel).cloned().unwrap_or_default();
            multi_segment |= segments.len() > 1;
            let reading = segments
                .iter()
                .find_map(|s| s.reading.as_ref().filter(|r| r.detected()))
                .or_else(|| segments.first().and_then(|s| s.reading.as_ref()))
                .cloned();
            let error = segments.iter().find_map(|s| s.error.clone());
            let decoder_sample_rate = match scheme {
                Decoder::AudioSeal => Some(if native_audioseal {
                    info.sample_rate
                } else {
                    decoders::audioseal::SAMPLE_RATE
                }),
                Decoder::SilentCipher => Some(decoders::silentcipher::SAMPLE_RATE),
                _ => None,
            };
            let inapplicable = (!segments.is_empty() || !windows.is_empty())
                && segments.iter().all(|s| s.inapplicable)
                && windows.iter().all(|w| w.inapplicable);
            AudioDetection {
                scheme,
                channel: *channel,
                status: status(reading.as_ref(), &windows, &segments),
                whole_clip: reading,
                windows,
                whole_clip_error: error,
                decoder_sample_rate,
                whole_segments: segments,
                inapplicable,
            }
        })
        .collect();
    let mut caveats = vec![format!(
        "{scheme} checks cover its pinned public model. Production-provider compatibility and general AI origin remain unverified."
    )];
    match scheme {
        Decoder::AudioSeal => caveats.push(
            "AudioSeal adds native-clock checks for calibrated 24 kHz and 48 kHz profiles alongside 16 kHz checks; \
             other native clocks are outside this coverage. Each channel and clock is independent. \
             A matching public-model mark does not identify LTX."
                .into(),
        ),
        Decoder::SilentCipher => {
            caveats.push("SilentCipher checks use the pinned 44.1 kHz model, including for 16 kHz input audio.".into());
        }
        _ => {}
    }
    if multi_segment {
        caveats.push(
            "Long-clip decoding used bounded overlapping segments alongside all sample windows.".into(),
        );
    }
    if detections.iter().any(|d| d.has_execution_error(false)) {
        caveats.push("Some decoder checks failed; those checks are unverified, not negative.".into());
    }
    (detections, caveats)
}

pub const OPENVOICE_NOTES: [&str; 2] = [
    "WavMark also checks the independently known OpenVoice default signature at its native sample offsets. \
     Custom messages and shifted/resampled framing are outside this receiver. \
     Provider identity and complete erasure remain unverified.",
    "Signature binomial tails are correlation diagnostics, not calibrated false-positive probabilities.",
];

/// `_eval_openvoice_frames`: the OpenVoice default signature on the native prefix.
pub fn eval_openvoice(
    decoders: &DecoderSet,
    frames: &Frames,
    info: &AudioStreamInfo,
) -> Result<Vec<AudioDetection>> {
    let net = decoders
        .wavmark
        .as_deref()
        .ok_or_else(|| missing(Decoder::WavMark))?;
    let mut checks = Vec::new();
    for channel in channels_of(info.channels) {
        let mono = mono_of(frames, channel);
        let base = AudioDetection {
            channel,
            decoder_sample_rate: Some(info.sample_rate),
            ..AudioDetection::missing(Decoder::WavMark)
        };
        if mono.len() < decoders::wavmark::MIN_READ_SAMPLES {
            checks.push(AudioDetection {
                whole_clip_error: Some(
                    "Audio is shorter than one complete OpenVoice signature window".into(),
                ),
                inapplicable: true,
                ..base
            });
            continue;
        }
        match decoders::wavmark::read_openvoice_default(net, &mono) {
            Err(err) => checks.push(AudioDetection {
                whole_clip_error: Some(err.to_string()),
                ..base
            }),
            Ok((whole, fragments)) => {
                let windows: Vec<AudioWindowReading> = fragments
                    .into_iter()
                    .map(|item| {
                        let start = item.start_samples[0];
                        AudioWindowReading::new(
                            start,
                            start + decoders::wavmark::MIN_READ_SAMPLES,
                            CheckOutcome::read(ModelReading::WavMarkSignature(item)),
                        )
                    })
                    .collect();
                let whole = ModelReading::WavMarkSignature(whole);
                checks.push(AudioDetection {
                    status: status(Some(&whole), &windows, &[]),
                    whole_clip: Some(whole),
                    windows,
                    ..base
                });
            }
        }
    }
    Ok(checks)
}

/// `_scan_audio_multi(source, info, schemes)`: one decode shared by every requested decoder.
pub fn scan_audio_multi(
    tools: &FfmpegTools,
    decoders: &DecoderSet,
    source: &Path,
    info: &AudioStreamInfo,
    schemes: &[Decoder],
) -> Result<(Vec<AudioDetection>, Vec<String>)> {
    if schemes.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if let Some(&absent) = schemes.iter().find(|&&s| !decoders.is_available(s)) {
        return Err(missing(absent));
    }
    let channels = channels_of(info.channels);
    let one_second: Vec<Decoder> = schemes
        .iter()
        .copied()
        .filter(|&s| s != Decoder::SilentCipher)
        .collect();
    let native_audioseal = one_second.contains(&Decoder::AudioSeal)
        && decoders::audioseal::NATIVE_SAMPLE_RATES.contains(&info.sample_rate);
    let has_openvoice = one_second.contains(&Decoder::WavMark);
    let has_silentcipher = schemes.contains(&Decoder::SilentCipher);
    let mut scanner = Scanner {
        decoders,
        info,
        lanes: HashMap::new(),
        silentcipher: Lane::new(&channels),
        channels: channels.clone(),
        one_second: one_second.clone(),
        native_audioseal,
    };
    for key in scanner.clock_keys() {
        scanner.lanes.insert(key, Lane::new(&channels));
    }
    let rate = info.sample_rate as usize;
    let win_1s = rate;
    let win_sc = rate * decoders::silentcipher::WINDOW_SECONDS;
    let mut acc_1s = (!one_second.is_empty())
        .then(|| WindowSegmentAccumulator::for_clock(info.channels, info.sample_rate, win_1s));
    let mut acc_sc = has_silentcipher
        .then(|| WindowSegmentAccumulator::for_clock(info.channels, info.sample_rate, win_sc));
    let read_chunk = if acc_sc.is_some() && acc_1s.is_none() && !has_openvoice {
        win_sc
    } else {
        win_1s
    };
    let mut sc_buf = Frames::empty(info.channels);
    let mut openvoice = Frames::empty(info.channels);
    let prefix = decoders::wavmark::OPENVOICE_PREFIX_SAMPLES;
    let mut events = Vec::new();
    for frames in AudioReader::open(tools, source, info, read_chunk)? {
        let frames = frames?;
        if !frames.is_finite() {
            return Err(AudioError::invalid("Audio input contains nonfinite samples"));
        }
        if has_openvoice && openvoice.frames() < prefix {
            openvoice.append(&frames.slice(0..prefix - openvoice.frames()));
        }
        if let Some(acc) = acc_1s.as_mut() {
            acc.feed(&frames, &mut events);
            for event in events.drain(..) {
                scanner.on_one_second(event)?;
            }
        }
        if let Some(acc) = acc_sc.as_mut() {
            if read_chunk == win_sc {
                acc.feed(&frames, &mut events);
            } else {
                let mut rest = frames;
                while !rest.is_empty() {
                    let take = rest.frames().min(win_sc - sc_buf.frames());
                    sc_buf.append(&rest.slice(0..take));
                    rest = rest.slice(take..rest.frames());
                    if sc_buf.frames() == win_sc {
                        let full = std::mem::replace(&mut sc_buf, Frames::empty(info.channels));
                        acc.feed(&full, &mut events);
                    }
                }
            }
            for event in events.drain(..) {
                scanner.on_silentcipher(event)?;
            }
        }
    }
    if let Some(acc) = acc_1s.as_mut() {
        acc.finish(&mut events);
        for event in events.drain(..) {
            scanner.on_one_second(event)?;
        }
    }
    if let Some(acc) = acc_sc.as_mut() {
        if !sc_buf.is_empty() {
            acc.feed(&sc_buf, &mut events);
        }
        acc.finish(&mut events);
        for event in events.drain(..) {
            scanner.on_silentcipher(event)?;
        }
    }

    let mut detections = Vec::new();
    let mut caveats = Vec::new();
    for &scheme in schemes {
        if scheme == Decoder::SilentCipher {
            let (dets, notes) = finalize(scheme, false, info, &channels, &scanner.silentcipher);
            detections.extend(dets);
            caveats.extend(notes);
            continue;
        }
        let empty = Lane::new(&channels);
        let lane = scanner.lanes.get(&(scheme, false)).unwrap_or(&empty);
        let (mut dets, mut notes) = finalize(scheme, false, info, &channels, lane);
        if scheme == Decoder::AudioSeal && native_audioseal {
            let lane = scanner.lanes.get(&(scheme, true)).unwrap_or(&empty);
            let (native, native_notes) = finalize(scheme, true, info, &channels, lane);
            dets.extend(native);
            for note in native_notes {
                if !notes.contains(&note) {
                    notes.push(note);
                }
            }
        } else if scheme == Decoder::WavMark {
            dets.extend(eval_openvoice(decoders, &openvoice, info)?);
            notes.extend(OPENVOICE_NOTES.iter().map(|&n| n.to_owned()));
        }
        detections.extend(dets);
        caveats.extend(notes);
    }
    Ok((detections, caveats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(events: &[Event]) -> Vec<(bool, usize, usize, bool)> {
        events
            .iter()
            .map(|e| match e {
                Event::Window { start, end, .. } => (true, *start, *end, false),
                Event::Segment { block, start, end } => (false, *start, *end, block.is_none()),
            })
            .collect()
    }

    #[test]
    fn short_tail_reuses_preceding_samples() {
        let mut acc = WindowSegmentAccumulator::new(1, 10, 100);
        let mut events = Vec::new();
        acc.feed(&Frames::zeros(10, 1), &mut events);
        acc.feed(&Frames::zeros(3, 1), &mut events);
        acc.finish(&mut events);
        assert_eq!(
            bounds(&events),
            vec![(true, 0, 10, false), (true, 3, 13, false), (false, 0, 13, false)]
        );
    }

    #[test]
    fn single_window_segment_is_reused() {
        let mut acc = WindowSegmentAccumulator::new(1, 10, 100);
        let mut events = Vec::new();
        acc.feed(&Frames::zeros(7, 1), &mut events);
        acc.finish(&mut events);
        assert_eq!(bounds(&events), vec![(true, 0, 7, false), (false, 0, 7, true)]);
    }

    #[test]
    fn long_clips_use_overlapping_segments() {
        let mut acc = WindowSegmentAccumulator::new(1, 10, 30);
        let mut events = Vec::new();
        for _ in 0..4 {
            acc.feed(&Frames::zeros(10, 1), &mut events);
        }
        acc.finish(&mut events);
        let segments: Vec<_> = bounds(&events).into_iter().filter(|e| !e.0).collect();
        assert_eq!(segments, vec![(false, 0, 30, false), (false, 20, 40, false)]);
    }
}
