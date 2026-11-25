//! Bounded neighboring context for independent-channel transforms (`_internal/audio_chunks.py`).

use crate::error::{AudioError, Result};
use crate::frames::Frames;

/// A fallible stream of sample blocks; dropping it closes the producer.
pub type ChunkStream<'a> = Box<dyn Iterator<Item = Result<Frames>> + Send + 'a>;

/// Applies a transform with input context, retaining exact central frame counts.
pub struct ContextTransform<I, F> {
    source: I,
    transform: F,
    context: usize,
    previous: Option<Frames>,
    current: Option<Frames>,
    started: bool,
    done: bool,
}

impl<I, F> std::fmt::Debug for ContextTransform<I, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextTransform")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

fn block(value: Frames) -> Result<Frames> {
    if value.frames() == 0 || value.channels() == 0 || !value.is_finite() {
        return Err(AudioError::invalid(
            "Audio chunks must contain finite nonempty sample frames",
        ));
    }
    Ok(value)
}

/// `transform_chunks(chunks, transform, context_frames=context)`.
pub fn transform_chunks<I, F>(source: I, transform: F, context: usize) -> Result<ContextTransform<I, F>>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    if context < 1 {
        return Err(AudioError::invalid(
            "Audio context must contain at least one frame",
        ));
    }
    Ok(ContextTransform {
        source,
        transform,
        context,
        previous: None,
        current: None,
        started: false,
        done: false,
    })
}

impl<I, F> ContextTransform<I, F>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    fn pull(&mut self) -> Result<Option<Frames>> {
        self.source.next().transpose()?.map(block).transpose()
    }

    fn step(&mut self) -> Result<Option<Frames>> {
        if !self.started {
            self.started = true;
            self.current = self.pull()?;
        }
        let Some(current) = self.current.take() else {
            return Ok(None);
        };
        let following = self.pull()?;
        if following.is_some() && current.frames() < self.context {
            return Err(AudioError::invalid(
                "Nonfinal audio chunks must contain enough transform context",
            ));
        }
        let previous = self
            .previous
            .take()
            .unwrap_or_else(|| Frames::empty(current.channels()));
        let mut padded = Frames::concat(&[previous.clone(), current.clone()], current.channels());
        if let Some(next) = &following {
            padded.append(&next.slice(0..self.context));
        }
        let result = (self.transform)(&padded)?;
        if result.channels() != padded.channels() || result.frames() != padded.frames() || !result.is_finite()
        {
            return Err(AudioError::failed(
                "Audio transform changed shape or returned nonfinite samples",
            ));
        }
        let start = previous.frames();
        let output = result.slice(start..start + current.frames());
        if let Some(next) = following {
            self.previous = Some(current.tail(self.context));
            self.current = Some(next);
        }
        Ok(Some(output))
    }
}

impl<I, F> Iterator for ContextTransform<I, F>
where
    I: Iterator<Item = Result<Frames>>,
    F: FnMut(&Frames) -> Result<Frames>,
{
    type Item = Result<Frames>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.step() {
            Ok(Some(frames)) => Some(Ok(frames)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(start: usize, len: usize) -> Frames {
        Frames::mono((start..start + len).map(|v| v as f32).collect())
    }

    #[test]
    fn central_frames_survive_with_context() {
        let chunks = vec![Ok(ramp(0, 5)), Ok(ramp(5, 5)), Ok(ramp(10, 2))];
        let mut seen = Vec::new();
        let out: Vec<Frames> = transform_chunks(
            chunks.into_iter(),
            |padded| {
                seen.push(padded.frames());
                Ok(padded.clone())
            },
            3,
        )
        .unwrap()
        .collect::<Result<_>>()
        .unwrap();
        assert_eq!(seen, vec![8, 10, 5]);
        assert_eq!(Frames::concat(&out, 1), ramp(0, 12));
    }

    #[test]
    fn short_nonfinal_chunk_is_rejected() {
        let chunks = vec![Ok(ramp(0, 2)), Ok(ramp(2, 2))];
        let mut stream = transform_chunks(chunks.into_iter(), |p| Ok(p.clone()), 4).unwrap();
        assert!(matches!(stream.next(), Some(Err(AudioError::Invalid(_)))));
        assert!(stream.next().is_none());
    }
}
