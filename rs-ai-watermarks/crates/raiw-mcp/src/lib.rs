//! MCP server and the integration operation catalog (port of `mcp_server.py` and `integration.py`).
//!
//! The CLI tree is the source of truth: [`catalog::operation_catalog`] walks the clap
//! command, and every tool call runs one CLI command in a child process, so a tool
//! returns exactly what that command would.

pub mod catalog;
pub mod exec;
pub mod server;
pub mod spec;
pub mod tools;

pub use catalog::{Operation, Parameter, operation_catalog};
pub use exec::{Completed, ProcessRunner, Runner};
pub use server::Server;
pub use spec::{BoundSpec, ToggleSpec};

use serde_json::{Map, Value};

/// JSON object arguments, as decoded from a request.
pub type Arguments = Map<String, Value>;

/// A request the integration refuses before anything runs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidArguments(pub String);

impl InvalidArguments {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
