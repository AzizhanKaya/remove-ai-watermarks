//! `urllib.parse.urlsplit` for the endpoint validation `text_api.py` performs.

/// The pieces of an absolute URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitUrl {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
    pub has_userinfo: bool,
    pub hostname: Option<String>,
    pub port: Option<u16>,
}

impl SplitUrl {
    /// `urlunsplit` without query and fragment, as `geturl()` yields for accepted URLs.
    pub fn base(&self) -> String {
        format!("{}://{}{}", self.scheme, self.netloc, self.path)
    }
}

/// Split `url`; `Err` mirrors the `ValueError` CPython raises for malformed ports or brackets.
pub(crate) fn split_url(url: &str) -> Result<SplitUrl, ()> {
    let (scheme, rest) = match url.split_once(':') {
        Some((scheme, rest))
            if !scheme.is_empty()
                && scheme.starts_with(|ch: char| ch.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.')) =>
        {
            (scheme.to_ascii_lowercase(), rest)
        }
        _ => (String::new(), url),
    };
    let (rest, fragment) = rest
        .split_once('#')
        .map_or((rest, ""), |(head, tail)| (head, tail));
    let (rest, query) = rest
        .split_once('?')
        .map_or((rest, ""), |(head, tail)| (head, tail));
    let (netloc, path) = match rest.strip_prefix("//") {
        Some(after) => {
            let end = after.find('/').unwrap_or(after.len());
            (&after[..end], &after[end..])
        }
        None => ("", rest),
    };
    if netloc.contains('[') != netloc.contains(']') {
        return Err(());
    }
    let (userinfo, hostport) = netloc
        .rsplit_once('@')
        .map_or((None, netloc), |(user, host)| (Some(user), host));
    let (host, port) = if let Some(bracketed) = hostport.strip_prefix('[') {
        let (host, after) = bracketed.split_once(']').ok_or(())?;
        (host.to_owned(), after.strip_prefix(':'))
    } else {
        match hostport.split_once(':') {
            Some((host, port)) => (host.to_owned(), Some(port)),
            None => (hostport.to_owned(), None),
        }
    };
    let port = match port {
        Some(port) if !port.is_empty() && port.chars().all(|ch| ch.is_ascii_digit()) => Some(
            port.parse::<u32>()
                .ok()
                .and_then(|value| u16::try_from(value).ok())
                .ok_or(())?,
        ),
        Some(_) => return Err(()),
        None => None,
    };
    Ok(SplitUrl {
        scheme,
        netloc: netloc.to_owned(),
        path: path.to_owned(),
        query: query.to_owned(),
        fragment: fragment.to_owned(),
        has_userinfo: userinfo.is_some(),
        hostname: (!host.is_empty()).then(|| host.to_ascii_lowercase()),
        port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_urlsplit() {
        let url = split_url("HTTP://127.0.0.1:11434/v1/").unwrap();
        assert_eq!(
            (
                url.scheme.as_str(),
                url.hostname.as_deref(),
                url.port,
                url.path.as_str()
            ),
            ("http", Some("127.0.0.1"), Some(11434), "/v1/")
        );
        let url = split_url("http://[::1]:8080/v1").unwrap();
        assert_eq!(url.hostname.as_deref(), Some("::1"));
        assert!(
            split_url("https://user:secret@example.test/v1")
                .unwrap()
                .has_userinfo
        );
        assert_eq!(
            split_url("https://example.test/v1?token=x").unwrap().query,
            "token=x"
        );
        assert!(split_url("http://host:99999/").is_err());
        assert!(split_url("http://host:/").is_err());
        assert_eq!(split_url("file:///tmp/model").unwrap().scheme, "file");
    }
}
