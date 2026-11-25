//! The bundled alpha and silhouette PNGs, embedded in place from the Python package.

use std::collections::HashMap;
use std::sync::OnceLock;

use raiw_cv::Mat;

use crate::error::{Result, VisibleError};

macro_rules! assets {
    ($($name:literal),* $(,)?) => {
        const ASSETS: &[(&str, &[u8])] = &[
            $(($name, include_bytes!(concat!("../../../../src/remove_ai_watermarks/assets/", $name))),)*
        ];
    };
}

assets!(
    "baidu_alpha.png",
    "dola_alpha.png",
    "doubao_alpha.png",
    "doubao_outline_alpha.png",
    "gemini_bg_48.png",
    "gemini_bg_96.png",
    "generic_ai_label_alpha.png",
    "genspark_alpha.png",
    "jimeng_alpha.png",
    "jimeng_label_alpha.png",
    "jimeng_pill.png",
    "kling_4_alpha.png",
    "kling_alpha.png",
    "kling_latin_4_alpha.png",
    "kling_latin_alpha.png",
    "liblib_alpha.png",
    "microsoft_alpha.png",
    "microsoft_text_alpha.png",
    "openart_alpha.png",
    "qwen_alpha.png",
    "qwen_symbol_alpha.png",
    "runninghub_alpha.png",
    "samsung_alpha.png",
    "samsung_ko_alpha.png",
    "wan_alpha.png",
    "workbuddy_alpha.png",
    "yuanbao_alpha.png",
    "yuanbao_compact_alpha.png",
    "yuanbao_compact_gray_alpha.png",
);

/// The raw PNG bytes of a bundled asset.
pub fn bytes(name: &str) -> Option<&'static [u8]> {
    ASSETS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// A decoded grayscale asset in its three cached forms.
#[derive(Debug)]
pub struct Template {
    /// `IMREAD_GRAYSCALE` pixels.
    pub gray: Mat<u8>,
    /// `gray / 255` as `float32`.
    pub alpha: Mat<f32>,
    /// `(alpha > 0.15) * 255`.
    pub silhouette: Mat<u8>,
}

fn decode_gray(name: &'static str, data: &[u8]) -> Result<Mat<u8>> {
    let img = image::load_from_memory_with_format(data, image::ImageFormat::Png)
        .map_err(|_| VisibleError::Asset(name))?
        .into_luma8();
    let (w, h) = img.dimensions();
    Ok(Mat::from_vec(w as usize, h as usize, 1, img.into_raw())?)
}

/// `IMREAD_COLOR` pixels of an RGB asset, as BGR.
pub fn decode_bgr(name: &'static str) -> Result<Mat<u8>> {
    let data = bytes(name).ok_or(VisibleError::Asset(name))?;
    let img = image::load_from_memory_with_format(data, image::ImageFormat::Png)
        .map_err(|_| VisibleError::Asset(name))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let bgr: Vec<u8> = img.pixels().flat_map(|p| [p[2], p[1], p[0]]).collect();
    Ok(Mat::from_vec(w as usize, h as usize, 3, bgr)?)
}

static TEMPLATES: OnceLock<HashMap<&'static str, Template>> = OnceLock::new();

fn templates() -> &'static HashMap<&'static str, Template> {
    TEMPLATES.get_or_init(|| {
        ASSETS
            .iter()
            .filter(|(name, _)| !name.starts_with("gemini_"))
            .filter_map(|&(name, data)| {
                let gray = decode_gray(name, data).ok()?;
                let alpha = gray.map(|v| f32::from(v) / 255.0);
                let silhouette = alpha.map(|v| if v > 0.15 { 255u8 } else { 0 });
                Some((name, Template { gray, alpha, silhouette }))
            })
            .collect()
    })
}

/// The decoded grayscale template for a bundled asset (`load_alpha_template` / `glyph_silhouette`).
pub fn template(name: &str) -> Result<&'static Template> {
    templates().get(name).ok_or_else(|| VisibleError::Unsupported {
        op: "template",
        reason: format!("no bundled asset {name:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_alpha_asset_decodes() {
        for (name, _) in ASSETS.iter().filter(|(n, _)| !n.starts_with("gemini_")) {
            let t = template(name).unwrap();
            assert!(t.gray.width() > 0 && t.gray.channels() == 1, "{name}");
        }
        assert_eq!(decode_bgr("gemini_bg_96.png").unwrap().size(), (96, 96));
    }
}
