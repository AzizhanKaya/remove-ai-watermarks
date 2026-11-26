//! `video` command handlers: arguments onto `raiw-video` options, results through its output helpers.

use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Result, anyhow};
use raiw_audio::options::{Device as VocoderDevice, Method, Scheme, Vendor};
use raiw_video::api::{
    AllVideoOptions, BatchVideoOptions, EraseVideoOptions, IdentifyVideoOptions, TrimBumper, VideoRuntime,
    VisibleVideoOptions,
};
use raiw_video::VideoAudioOptions;
use raiw_video::cli::{self as out, CommandOutput};
use raiw_video::invisible::{InvisibleMode, InvisibleOptions};
use raiw_video::messages::{
    MessageRemovalOptions, MessageScheme, detect_video_watermark_presence, identify_video_message,
    remove_video_message,
};
use raiw_video::report::BatchMode;
use raiw_visible::{Backend, Region};

use super::args::{AudioProcessArgs, AudioVendor, FillBackend, VideoAudioArgs, VideoRegenArgs};
use super::video::{
    MessageCmd, VideoAllCmd, VideoAudioCmd, VideoBatchCmd, VideoCmd, VideoEraseCmd, VideoIdentifyCmd,
    VideoInvisibleCmd, VideoMetadataCmd, VideoVisibleArgs, VideoVisibleCmd,
};

/// Exit code of a Click usage error raised inside a command (`click.BadParameter`).
const EXIT_USAGE: i32 = 2;

pub fn video(command: VideoCmd) -> Result<i32> {
    match command {
        VideoCmd::Identify(cmd) => identify(&cmd),
        VideoCmd::Metadata(cmd) => metadata(&cmd),
        VideoCmd::Visible(cmd) => visible(&cmd),
        VideoCmd::Erase(cmd) => erase(&cmd),
        VideoCmd::Invisible(cmd) => invisible(&cmd),
        VideoCmd::All(cmd) => all(&cmd),
        VideoCmd::Batch(cmd) => batch(&cmd),
        VideoCmd::Audio(cmd) => audio(&cmd),
        VideoCmd::Message(cmd) => message(cmd),
    }
}

fn message(command: MessageCmd) -> Result<i32> {
    match command {
        MessageCmd::Identify(cmd) => {
            let runtime = VideoRuntime::discover();
            let scheme = MessageScheme::parse(cmd.scheme.as_str())?;
            let report = identify_video_message(&runtime, &cmd.source, &cmd.reference, scheme, true);
            verified(report, cmd.as_json, |r| out::message_identify_output(r, cmd.as_json))
        }
        MessageCmd::Detect(cmd) => {
            let runtime = VideoRuntime::discover();
            let scheme = MessageScheme::parse(cmd.scheme.as_str())?;
            let report = detect_video_watermark_presence(&runtime, &cmd.source, scheme);
            verified(report, cmd.as_json, |r| out::message_detect_output(r, cmd.as_json))
        }
        MessageCmd::Remove(cmd) => {
            let runtime = VideoRuntime::discover_on(cmd.device.as_str())?;
            let options = MessageRemovalOptions {
                reference_hex: cmd.reference.clone(),
                scheme: MessageScheme::parse(cmd.scheme.as_str())?,
                method: cmd.method.as_str().to_owned(),
                noise_std: cmd.noise_std,
                long_side: cmd.long_side as usize,
                device: cmd.device.as_str().to_owned(),
                force: cmd.force,
                verify: cmd.verify.get(),
                ..MessageRemovalOptions::default()
            };
            let output = cmd.output.clone().unwrap_or_else(|| out::message_default_output(&cmd.source));
            let result = remove_video_message(&runtime, &cmd.source, &output, &options);
            verified(result, cmd.as_json, |r| out::message_remove_output(r, cmd.as_json))
        }
    }
}

/// `_verification_error`: a JSON error report under `--json`, a Click error otherwise.
fn verified<T>(result: raiw_video::Result<T>, as_json: bool, render: impl FnOnce(&T) -> CommandOutput) -> Result<i32> {
    match result {
        Ok(value) => Ok(emit(&render(&value))),
        Err(err) if as_json => Ok(emit(&out::verification_error_json(&err))),
        Err(err) => Err(anyhow!("{err}")),
    }
}

// Shared plumbing

fn banner() {
    println!("{}", raiw_metadata::report::banner(crate::VERSION));
}

fn emit(output: &CommandOutput) -> i32 {
    for line in &output.lines {
        println!("{line}");
    }
    output.exit_code
}

fn parse<T: FromStr>(text: &str) -> Result<T>
where
    T::Err: Display,
{
    text.parse().map_err(|err| anyhow!("{err}"))
}

fn backend(choice: FillBackend) -> Result<Backend> {
    Backend::parse(choice.as_str()).ok_or_else(|| anyhow!("unknown backend {choice}"))
}

fn vendor(choice: AudioVendor) -> Result<Option<Vendor>> {
    match choice {
        AudioVendor::Auto => Ok(None),
        other => parse(other.as_str()).map(Some),
    }
}

fn usage_error(message: &str) -> i32 {
    eprintln!("Error: Invalid value: {message}");
    EXIT_USAGE
}

fn visible_options(args: &VideoVisibleArgs, temporal_consistency: bool, trim_bumper: bool) -> Result<VisibleVideoOptions> {
    Ok(VisibleVideoOptions {
        mark: args.mark.as_str().to_owned(),
        backend: backend(args.backend)?,
        temporal_consistency,
        trim_bumper: if trim_bumper { TrimBumper::On } else { TrimBumper::Off },
        ..VisibleVideoOptions::default()
    })
}

fn invisible_options(regen: &VideoRegenArgs) -> Result<InvisibleOptions> {
    Ok(InvisibleOptions {
        profile: regen.profile.as_str().to_owned(),
        quality: regen.quality.as_str().to_owned(),
        noise_std: regen.noise_std,
        long_side: regen.long_side as usize,
        fps: regen.effective_fps(),
        batch_size: regen.batch_size as usize,
        seed: u64::try_from(regen.seed).map_err(|_| anyhow!("--seed must be non-negative"))?,
        device: regen.device.as_str().to_owned(),
        temporal_stabilize: regen.temporal_stabilize.get(),
        native_guide: regen.native_guide,
        wan_steps: regen.wan_steps,
        ..InvisibleOptions::default()
    })
}

fn audio_options(process: &AudioProcessArgs, scheme: &str) -> Result<VideoAudioOptions> {
    Ok(VideoAudioOptions {
        force: process.force,
        verify: process.verify.get(),
        method: parse::<Method>(process.method.as_str())?,
        bandwidth_kbps: f64::from(process.bandwidth.kbps()),
        scheme: parse::<Scheme>(scheme)?,
        vocoder_device: parse::<VocoderDevice>(process.vocoder_device.as_str())?,
        allow_unverified: process.allow_unverified,
        allow_positive: process.allow_positive,
        allow_quality_override: process.allow_quality_override,
        vendor: vendor(process.vendor)?,
        ..VideoAudioOptions::default()
    })
}

fn embedded_audio_options(args: &VideoAudioArgs) -> Result<VideoAudioOptions> {
    let streams: Vec<usize> = args.audio_streams.iter().map(|s| *s as usize).collect();
    Ok(VideoAudioOptions {
        force: args.audio_force,
        method: parse::<Method>(args.audio_method.as_str())?,
        scheme: parse::<Scheme>(args.audio_scheme.as_str())?,
        allow_unverified: args.audio_allow_unverified,
        allow_positive: args.audio_allow_positive,
        allow_quality_override: args.audio_allow_quality_override,
        vendor: vendor(args.audio_vendor)?,
        streams: (!streams.is_empty()).then_some(streams),
        ..VideoAudioOptions::default()
    })
}

fn all_options(cmd_visible: VisibleVideoOptions, invisible: bool, regen: &VideoRegenArgs, audio: &VideoAudioArgs) -> Result<AllVideoOptions> {
    Ok(AllVideoOptions {
        visible: cmd_visible,
        include_invisible: if invisible { InvisibleMode::On } else { InvisibleMode::Off },
        invisible: invisible_options(regen)?,
        include_audio_invisible: audio.audio_invisible.get(),
        audio: embedded_audio_options(audio)?,
        ..AllVideoOptions::default()
    })
}

fn parse_region(spec: &str) -> std::result::Result<Region, String> {
    let compact = spec.replace(' ', "");
    let parts: Vec<&str> = compact.split(',').collect();
    if parts.len() != 4 {
        return Err(format!("region must be 'x,y,w,h', got: '{spec}'"));
    }
    let values: Vec<i32> = parts
        .iter()
        .map(|p| p.parse::<i32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| format!("region values must be integers: '{spec}'"))?;
    let [x, y, w, h] = values[..] else { unreachable!("four parts were checked") };
    if w <= 0 || h <= 0 {
        return Err(format!("region width/height must be positive: '{spec}'"));
    }
    Ok(Region { x, y, w, h })
}

fn parse_frame_range(spec: Option<&str>) -> std::result::Result<Option<(usize, usize)>, String> {
    let Some(spec) = spec.filter(|s| !s.trim().is_empty()) else { return Ok(None) };
    let parts: Vec<&str> = spec.trim().split(':').collect();
    if parts.len() != 2 {
        return Err(format!("frames must be 'START:END' (0-based inclusive), got: '{spec}'"));
    }
    let (start, end) = match (parts[0].trim().parse::<i64>(), parts[1].trim().parse::<i64>()) {
        (Ok(start), Ok(end)) => (start, end),
        _ => return Err(format!("frame indices must be integers: '{spec}'")),
    };
    if start < 0 || end < start {
        return Err(format!("frame range must satisfy 0 <= START <= END: '{spec}'"));
    }
    Ok(Some((start as usize, end as usize)))
}

// Commands

fn identify(cmd: &VideoIdentifyCmd) -> Result<i32> {
    let options = IdentifyVideoOptions {
        check_visible: !cmd.no_visible,
        check_audio: cmd.audio.get(),
        audio_scheme: parse(cmd.audio_scheme.as_str())?,
    };
    let report = raiw_video::api::identify_video(&VideoRuntime::discover(), &cmd.source, &options)?;
    if cmd.as_json {
        return Ok(emit(&out::identify_json(&report)));
    }
    banner();
    Ok(emit(&out::identify_text(&report, cmd.no_visible, cmd.audio.get())))
}

fn metadata(cmd: &VideoMetadataCmd) -> Result<i32> {
    let (check, remove, as_json) = (cmd.mode.check, cmd.mode.remove, cmd.as_json);
    if !as_json {
        banner();
    }
    if check || !remove {
        let report = raiw_video::api::inspect_video_metadata(&cmd.source)?;
        if as_json && !remove {
            return Ok(emit(&out::metadata_check_json(&report)));
        }
        if !as_json {
            print!(
                "{}",
                raiw_metadata::report::metadata_report(&cmd.source, report.has_ai_metadata, &report.markers)
            );
        }
        if !remove {
            return Ok(0);
        }
    }
    let result = raiw_video::api::remove_video_metadata(
        &cmd.source,
        cmd.output.as_deref(),
        cmd.mode.keep_standard.get(),
        None,
    )?;
    Ok(emit(&out::metadata_remove_output(&result, as_json)))
}

fn visible(cmd: &VideoVisibleCmd) -> Result<i32> {
    let mark = cmd.visible.mark.as_str();
    if !cmd.as_json {
        banner();
        println!("{}", out::visible_start_line(&cmd.source, mark));
    }
    let mut options = visible_options(&cmd.visible, cmd.temporal_consistency.get(), cmd.trim_bumper)?;
    options.strip_metadata = cmd.strip_metadata.get();
    let result =
        raiw_video::api::remove_video_visible(&VideoRuntime::discover(), &cmd.source, cmd.output.as_deref(), &options)?;
    Ok(emit(&out::visible_output(&result, mark, cmd.as_json)))
}

fn erase(cmd: &VideoEraseCmd) -> Result<i32> {
    let regions = match cmd.regions.iter().map(|r| parse_region(r)).collect::<std::result::Result<Vec<_>, _>>() {
        Ok(regions) => regions,
        Err(message) => return Ok(usage_error(&message)),
    };
    let frame_range = match parse_frame_range(cmd.frames.as_deref()) {
        Ok(range) => range,
        Err(message) => return Ok(usage_error(&message)),
    };
    if !cmd.as_json {
        banner();
        println!("{}", out::erase_start_line(&cmd.source, regions.len()));
    }
    let options = EraseVideoOptions {
        regions,
        frame_range,
        backend: backend(cmd.backend)?,
        dilate: i32::try_from(cmd.dilate).unwrap_or(i32::MAX),
        strip_metadata: cmd.strip_metadata.get(),
        temporal_consistency: cmd.temporal_consistency.get(),
        track: cmd.track,
    };
    let result =
        raiw_video::api::remove_video_region(&VideoRuntime::discover(), &cmd.source, cmd.output.as_deref(), &options)?;
    Ok(emit(&out::erase_output(&result, cmd.as_json)))
}

fn invisible(cmd: &VideoInvisibleCmd) -> Result<i32> {
    let options = invisible_options(&cmd.regen)?;
    if !cmd.as_json {
        banner();
        println!("  Regenerating {} with the {} profile...", file_name(&cmd.source), options.profile);
    }
    let result = raiw_video::api::remove_video_invisible(
        &VideoRuntime::discover_on(cmd.regen.device.as_str())?,
        &cmd.source,
        cmd.output.as_deref(),
        &options,
    )?;
    Ok(emit(&out::invisible_output(&result, cmd.as_json)))
}

fn all(cmd: &VideoAllCmd) -> Result<i32> {
    let visible = visible_options(&cmd.visible, cmd.temporal_consistency.get(), cmd.trim_bumper)?;
    let options = all_options(visible, cmd.invisible.get(), &cmd.regen, &cmd.audio)?;
    let profile = cmd.regen.profile.as_str();
    if !cmd.as_json {
        banner();
        let audio_stage = cmd.audio.audio_invisible.get() || cmd.audio.audio_vendor != AudioVendor::Auto;
        let scheme = cmd.audio.audio_scheme.as_str();
        println!(
            "{}",
            out::all_start_line(&cmd.source, cmd.invisible.get().then_some(profile), audio_stage.then_some(scheme))
        );
    }
    let runtime = VideoRuntime::discover_on(cmd.regen.device.as_str())?;
    let result = raiw_video::api::remove_video_all(&runtime, &cmd.source, cmd.output.as_deref(), &options)?;
    Ok(emit(&out::all_output(&result, profile, cmd.as_json)))
}

fn batch(cmd: &VideoBatchCmd) -> Result<i32> {
    let mode = BatchMode::parse(cmd.mode.as_str()).ok_or_else(|| anyhow!("unknown mode {}", cmd.mode))?;
    let visible = visible_options(&cmd.visible, cmd.temporal_consistency.get(), false)?;
    let options = BatchVideoOptions { mode, all: all_options(visible, cmd.invisible.get(), &cmd.regen, &cmd.audio)? };
    if !cmd.as_json {
        banner();
        println!("{}", out::batch_start_line(&cmd.directory, cmd.mode.as_str()));
    }
    let result = raiw_video::api::remove_video_batch(
        &VideoRuntime::discover_on(cmd.regen.device.as_str())?,
        &cmd.directory,
        cmd.output_dir.as_deref(),
        &options,
    )?;
    Ok(emit(&out::batch_output(&result, &cmd.directory, cmd.audio.audio_invisible.get(), cmd.as_json)))
}

fn audio(cmd: &VideoAudioCmd) -> Result<i32> {
    let scheme = cmd.scheme.as_str();
    let mut options = audio_options(&cmd.process, scheme)?;
    let target = cmd
        .output
        .clone()
        .unwrap_or_else(|| raiw_core::fs::with_stem_replaced(&cmd.source, "_audio_processed.mkv"));
    let runtime = VideoRuntime::discover();
    let multi = cmd.all_streams || !cmd.target_streams.is_empty();
    let outcome = if multi {
        if !cmd.target_streams.is_empty() {
            options.streams = Some(cmd.target_streams.iter().map(|s| *s as usize).collect());
        }
        raiw_video::remove_video_audio_all_streams(&runtime.audio, &cmd.source, &target, &options, None)
            .map(|result| out::audio_multi_output(&result, scheme, cmd.as_json))
    } else {
        raiw_video::remove_video_audio(&runtime.audio, &cmd.source, &target, cmd.stream as usize, &options, None)
            .map(|result| out::audio_single_output(&result, cmd.stream as usize, scheme, cmd.as_json))
    };
    verified(outcome, cmd.as_json, Clone::clone)
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_and_frames_follow_the_python_parsers() {
        assert_eq!(parse_region(" 1, 2,3,4").unwrap(), Region { x: 1, y: 2, w: 3, h: 4 });
        assert_eq!(parse_region("1,2,3").unwrap_err(), "region must be 'x,y,w,h', got: '1,2,3'");
        assert!(parse_region("1,2,0,4").unwrap_err().contains("positive"));
        assert!(parse_region("a,2,3,4").unwrap_err().contains("integers"));
        assert_eq!(parse_frame_range(Some(" 3:7 ")).unwrap(), Some((3, 7)));
        assert_eq!(parse_frame_range(Some("  ")).unwrap(), None);
        assert!(parse_frame_range(Some("7:3")).unwrap_err().contains("0 <= START <= END"));
    }
}
