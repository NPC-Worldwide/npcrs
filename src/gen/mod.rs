pub mod response_types;

use std::sync::atomic::AtomicBool;

/// Set by `npcrs_cancel_inference` to abort an in-flight local decode. The
/// llamacpp token loop checks it every step; FFI entry points clear it when a
/// turn begins. Lives here (not in the feature-gated module) so the FFI layer
/// can touch it regardless of build features.
pub static INFERENCE_CANCELLED: AtomicBool = AtomicBool::new(false);

pub mod audio_gen;
pub mod cost;
pub mod embeddings;
mod image_gen;
#[cfg(feature = "llamacpp")]
pub mod llamacpp;
pub mod model_info;
pub mod response;
pub mod sanitize;

pub use cost::calculate_cost;
pub use image_gen::*;
#[cfg(feature = "llamacpp")]
pub use llamacpp::get_llamacpp_response;
pub use response::get_genai_response;
pub use response_types::*;
pub use sanitize::sanitize_messages;
