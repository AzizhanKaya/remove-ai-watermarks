use sha2::{Digest, Sha256};

#[test]
#[ignore = "probe"]
fn probe_decodes() {
    let list = std::fs::read_to_string(
        "/tmp/claude-1000/-opt-remove-ai-watermarks/e1a1741c-245d-448c-ab88-df5599173e44/scratchpad/cv2_decode.txt",
    )
    .unwrap();
    for line in list.lines() {
        let mut parts = line.split_whitespace();
        let rel = parts.next().unwrap();
        let expected = line.split_whitespace().nth(4).unwrap();
        let path = raiw_core::testing::repo_root().join(rel);
        let img = image::open(&path).unwrap();
        let bytes: Vec<u8> = match &img {
            image::DynamicImage::ImageRgb8(b) => b.pixels().flat_map(|p| [p[2], p[1], p[0]]).collect(),
            image::DynamicImage::ImageRgba8(b) => b.pixels().flat_map(|p| [p[2], p[1], p[0], p[3]]).collect(),
            other => panic!("{rel}: {:?}", other.color()),
        };
        let got = hex::encode(Sha256::digest(&bytes));
        println!("{} {} {}", if got.starts_with(expected) { "OK " } else { "BAD" }, rel, &got[..16]);
    }
}
