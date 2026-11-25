//! Minimal Hugging Face hub client over the standard `huggingface_hub` cache layout.
//!
//! Files resolve to `<cache>/models--<org>--<name>/snapshots/<commit>/<filename>`, blobs are
//! shared under `blobs/<etag>`, and snapshots the Python package already fetched are reused.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use raiw_core::fs::home_dir;
use raiw_core::sha256_file;

use crate::error::{GpuError, Result};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const USER_AGENT: &str = concat!("raiw-gpu/", env!("CARGO_PKG_VERSION"));

/// One pinned file of a model repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HubFile<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    pub filename: &'a str,
    /// Expected SHA-256 of the file contents, when the Python code pins one.
    pub sha256: Option<&'a str>,
}

impl<'a> HubFile<'a> {
    pub const fn new(repo: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            repo,
            revision,
            filename,
            sha256: None,
        }
    }

    #[must_use]
    pub const fn with_sha256(mut self, sha256: &'a str) -> Self {
        self.sha256 = Some(sha256);
        self
    }
}

/// A file bundled inside a pinned remote zip archive (e.g. a checkpoint shipped in a PyPI wheel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchiveMember<'a> {
    pub url: &'a str,
    pub archive_sha256: &'a str,
    pub member: &'a str,
    pub sha256: &'a str,
    /// Path under `raiw_core::model_cache_dir()` where the extracted member persists.
    pub cache_path: &'a str,
}

/// Whether a missing file may be fetched (Python's `local_files_only`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fetch {
    CacheOnly,
    #[default]
    Download,
}

/// Hub cache plus the network settings `huggingface_hub` reads from the environment.
#[derive(Debug, Clone)]
pub struct Hub {
    cache: PathBuf,
    endpoint: String,
    offline: bool,
    token: Option<String>,
}

impl Default for Hub {
    fn default() -> Self {
        Self::from_env()
    }
}

impl Hub {
    /// Honor `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, `HF_ENDPOINT`, `HF_HUB_OFFLINE` and `HF_TOKEN`.
    pub fn from_env() -> Self {
        Self {
            cache: default_cache_dir(),
            endpoint: env_value("HF_ENDPOINT").unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned()),
            offline: env_value("HF_HUB_OFFLINE").is_some_and(|v| is_truthy(&v)),
            token: env_value("HF_TOKEN"),
        }
    }

    /// A hub rooted at an explicit cache directory (tests, sandboxes).
    pub fn with_cache(cache: impl Into<PathBuf>) -> Self {
        Self {
            cache: cache.into(),
            ..Self::from_env()
        }
    }

    #[must_use]
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    #[must_use]
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache
    }

    pub fn repo_dir(&self, repo: &str) -> PathBuf {
        self.cache.join(format!("models--{}", repo.replace('/', "--")))
    }

    /// Cached snapshot path of a file, without touching the network or checking digests.
    pub fn cached(&self, file: &HubFile<'_>) -> Option<PathBuf> {
        let repo_dir = self.repo_dir(file.repo);
        let commit = resolve_commit(&repo_dir, file.revision)?;
        let path = repo_dir.join("snapshots").join(commit).join(file.filename);
        path.is_file().then_some(path)
    }

    /// Resolve a file to a local path, downloading it into the cache when allowed.
    pub fn resolve(&self, file: &HubFile<'_>, fetch: Fetch) -> Result<PathBuf> {
        if let Some(path) = self.cached(file) {
            verify_digest(file, &path)?;
            return Ok(path);
        }
        if self.offline || fetch == Fetch::CacheOnly {
            return Err(GpuError::NotCached {
                repo: file.repo.to_owned(),
                revision: file.revision.to_owned(),
                filename: file.filename.to_owned(),
            });
        }
        self.download(file)
    }

    /// Resolve an archive member into the model cache, downloading and extracting it when allowed.
    pub fn resolve_archive_member(&self, member: &ArchiveMember<'_>, fetch: Fetch) -> Result<PathBuf> {
        let target = raiw_core::model_cache_dir().join(member.cache_path);
        let pinned = HubFile::new(member.url, "", member.member).with_sha256(member.sha256);
        if target.is_file() {
            verify_digest(&pinned, &target)?;
            return Ok(target);
        }
        if self.offline || fetch == Fetch::CacheOnly {
            return Err(GpuError::NotCached {
                repo: member.url.to_owned(),
                revision: member.archive_sha256.to_owned(),
                filename: member.member.to_owned(),
            });
        }
        let archive = target.with_extension("archive.incomplete");
        self.fetch_blob(
            member.url,
            &archive,
            &HubFile::new(member.url, "", member.url).with_sha256(member.archive_sha256),
        )?;
        let extracted = extract_member(&archive, member.member, &target);
        let _ = fs::remove_file(&archive);
        extracted?;
        if let Err(err) = verify_digest(&pinned, &target) {
            let _ = fs::remove_file(&target);
            return Err(err);
        }
        Ok(target)
    }

    fn download(&self, file: &HubFile<'_>) -> Result<PathBuf> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint.trim_end_matches('/'),
            file.repo,
            file.revision,
            file.filename
        );
        let meta = self.head(&url)?;
        let commit = meta.commit.unwrap_or_else(|| file.revision.to_owned());
        let repo_dir = self.repo_dir(file.repo);
        let blob = repo_dir.join("blobs").join(&meta.etag);
        if !blob.is_file() {
            let source = meta.location.as_deref().unwrap_or(&url);
            self.fetch_blob(source, &blob, file)?;
        }
        let snapshot = repo_dir.join("snapshots").join(&commit).join(file.filename);
        link_snapshot(&blob, &snapshot)?;
        if file.revision != commit {
            let refs = repo_dir.join("refs").join(file.revision);
            create_parent(&refs)?;
            fs::write(&refs, &commit).map_err(|err| GpuError::io(&refs, err))?;
        }
        verify_digest(file, &snapshot)?;
        Ok(snapshot)
    }

    fn agent(&self, redirects: u32) -> ureq::Agent {
        ureq::Agent::config_builder()
            .max_redirects(redirects)
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .user_agent(USER_AGENT)
            .build()
            .into()
    }

    fn head(&self, url: &str) -> Result<FileMeta> {
        let mut request = self.agent(0).head(url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = request.call().map_err(|err| http_error(url, &err))?;
        let status = response.status().as_u16();
        if !(200..400).contains(&status) {
            return Err(GpuError::Http {
                url: url.to_owned(),
                reason: format!("HTTP {status}"),
            });
        }
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let etag = header("x-linked-etag")
            .or_else(|| header("etag"))
            .map(|tag| normalize_etag(&tag))
            .filter(|tag| !tag.is_empty() && !tag.contains(['/', '\\']))
            .ok_or_else(|| GpuError::Http {
                url: url.to_owned(),
                reason: "response carries no ETag".to_owned(),
            })?;
        let location = header("location").map(|loc| absolute_location(url, &loc));
        Ok(FileMeta {
            commit: header("x-repo-commit"),
            etag,
            location,
        })
    }

    fn fetch_blob(&self, url: &str, blob: &Path, file: &HubFile<'_>) -> Result<()> {
        create_parent(blob)?;
        let partial = blob.with_extension("incomplete");
        let mut request = self.agent(10).get(url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = request.call().map_err(|err| http_error(url, &err))?;
        let status = response.status().as_u16();
        if status != 200 {
            return Err(GpuError::Http {
                url: url.to_owned(),
                reason: format!("HTTP {status}"),
            });
        }
        let mut out = fs::File::create(&partial).map_err(|err| GpuError::io(&partial, err))?;
        io::copy(&mut response.into_body().into_reader(), &mut out)
            .map_err(|err| GpuError::io(&partial, err))?;
        out.sync_all().map_err(|err| GpuError::io(&partial, err))?;
        drop(out);
        if let Err(err) = verify_digest(file, &partial) {
            let _ = fs::remove_file(&partial);
            return Err(err);
        }
        fs::rename(&partial, blob).map_err(|err| GpuError::io(blob, err))
    }
}

struct FileMeta {
    commit: Option<String>,
    etag: String,
    location: Option<String>,
}

fn default_cache_dir() -> PathBuf {
    if let Some(cache) = env_value("HF_HUB_CACHE") {
        return expand_home(&cache);
    }
    if let Some(home) = env_value("HF_HOME") {
        return expand_home(&home).join("hub");
    }
    let base = env_value("XDG_CACHE_HOME").map_or_else(|| home_dir().join(".cache"), |xdg| expand_home(&xdg));
    base.join("huggingface").join("hub")
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "on" | "yes" | "true"
    )
}

fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix("~/").or_else(|| (path == "~").then_some("")) {
        Some(rest) => home_dir().join(rest),
        None => PathBuf::from(path),
    }
}

fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

fn resolve_commit(repo_dir: &Path, revision: &str) -> Option<String> {
    if is_commit_hash(revision) {
        return Some(revision.to_owned());
    }
    let commit = fs::read_to_string(repo_dir.join("refs").join(revision)).ok()?;
    let commit = commit.trim();
    is_commit_hash(commit).then(|| commit.to_owned())
}

fn normalize_etag(tag: &str) -> String {
    tag.trim().trim_start_matches("W/").trim_matches('"').to_owned()
}

fn absolute_location(url: &str, location: &str) -> String {
    if location.starts_with("http://") || location.starts_with("https://") {
        return location.to_owned();
    }
    let origin_end = url.find("://").map_or(0, |i| i + 3);
    let host_end = url[origin_end..].find('/').map_or(url.len(), |i| origin_end + i);
    format!("{}{}", &url[..host_end], location)
}

fn verify_digest(file: &HubFile<'_>, path: &Path) -> Result<()> {
    let Some(expected) = file.sha256 else {
        return Ok(());
    };
    let actual = sha256_file(path).map_err(|err| GpuError::io(path, err))?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(GpuError::DigestMismatch {
            filename: file.filename.to_owned(),
            actual,
            expected: expected.to_owned(),
        })
    }
}

fn extract_member(archive: &Path, member: &str, target: &Path) -> Result<()> {
    let fail = |reason: String| GpuError::Weights {
        path: archive.to_path_buf(),
        reason,
    };
    let file = fs::File::open(archive).map_err(|err| GpuError::io(archive, err))?;
    let mut zip = zip::ZipArchive::new(io::BufReader::new(file)).map_err(|err| fail(err.to_string()))?;
    let mut entry = zip
        .by_name(member)
        .map_err(|err| fail(format!("{member}: {err}")))?;
    create_parent(target)?;
    let partial = target.with_extension("incomplete");
    let mut out = fs::File::create(&partial).map_err(|err| GpuError::io(&partial, err))?;
    io::copy(&mut entry, &mut out).map_err(|err| GpuError::io(&partial, err))?;
    drop(out);
    fs::rename(&partial, target).map_err(|err| GpuError::io(target, err))
}

fn create_parent(path: &Path) -> Result<()> {
    match path.parent() {
        Some(parent) => fs::create_dir_all(parent).map_err(|err| GpuError::io(parent, err)),
        None => Ok(()),
    }
}

fn link_snapshot(blob: &Path, snapshot: &Path) -> Result<()> {
    create_parent(snapshot)?;
    if snapshot.exists() {
        return Ok(());
    }
    link_or_copy(blob, snapshot).map_err(|err| GpuError::io(snapshot, err))
}

#[cfg(unix)]
fn link_or_copy(blob: &Path, snapshot: &Path) -> io::Result<()> {
    let depth = snapshot
        .parent()
        .and_then(|dir| dir.strip_prefix(blob.parent()?.parent()?).ok());
    let relative = match (depth, blob.file_name()) {
        (Some(depth), Some(name)) => {
            let ups: PathBuf = depth.components().map(|_| "..").collect();
            ups.join("blobs").join(name)
        }
        _ => blob.to_path_buf(),
    };
    std::os::unix::fs::symlink(relative, snapshot).or_else(|_| fs::copy(blob, snapshot).map(drop))
}

#[cfg(not(unix))]
fn link_or_copy(blob: &Path, snapshot: &Path) -> io::Result<()> {
    fs::copy(blob, snapshot).map(drop)
}

fn http_error(url: &str, err: &ureq::Error) -> GpuError {
    GpuError::Http {
        url: url.to_owned(),
        reason: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "c1bc521685adf9cfe247bc39a5ca58917eda1ac4";

    fn seeded_cache(dir: &Path, contents: &[u8]) -> Hub {
        let repo = dir.join("models--org--model");
        let blob = repo.join("blobs").join("abc123");
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, contents).unwrap();
        link_snapshot(
            &blob,
            &repo.join("snapshots").join(COMMIT).join("sub/config.json"),
        )
        .unwrap();
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::write(repo.join("refs").join("main"), COMMIT).unwrap();
        Hub::with_cache(dir).offline(true)
    }

    #[test]
    fn reuses_cached_snapshot_by_commit_and_ref() {
        let dir = tempfile::tempdir().unwrap();
        let hub = seeded_cache(dir.path(), b"{}");
        for revision in [COMMIT, "main"] {
            let file = HubFile::new("org/model", revision, "sub/config.json");
            let path = hub.resolve(&file, Fetch::Download).unwrap();
            assert_eq!(fs::read(path).unwrap(), b"{}");
        }
    }

    #[test]
    fn verifies_pinned_digest() {
        let dir = tempfile::tempdir().unwrap();
        let hub = seeded_cache(dir.path(), b"abc");
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let good = HubFile::new("org/model", COMMIT, "sub/config.json").with_sha256(digest);
        assert!(hub.resolve(&good, Fetch::CacheOnly).is_ok());
        let bad = good.with_sha256("00");
        assert!(matches!(
            hub.resolve(&bad, Fetch::CacheOnly),
            Err(GpuError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn offline_miss_is_not_cached_error() {
        let dir = tempfile::tempdir().unwrap();
        let hub = Hub::with_cache(dir.path()).offline(true);
        let file = HubFile::new("org/missing", COMMIT, "model.bin");
        assert!(matches!(
            hub.resolve(&file, Fetch::Download),
            Err(GpuError::NotCached { .. })
        ));
    }

    #[test]
    fn etag_and_location_normalization() {
        assert_eq!(normalize_etag("W/\"abc\""), "abc");
        assert_eq!(
            absolute_location("https://huggingface.co/a/resolve/x/f", "/api/x"),
            "https://huggingface.co/api/x"
        );
        assert_eq!(
            absolute_location("https://h.co/a", "https://cdn.example/f"),
            "https://cdn.example/f"
        );
    }
}
