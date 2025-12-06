//! Configuration loading and management

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub processing: ProcessingConfig,
    pub output: OutputConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub backend: Backend,
    pub quantization: Quantization,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Transformers,
    Vllm,
    Ollama,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    None,
    Int8,
    Int4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub archive_dir: PathBuf,
    pub archive_processed: bool,
    pub image_dpi: u32,
    pub image_format: ImageFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub include_raw_text: bool,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Csv,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: u32,
    pub temperature: f64,
    pub server_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                name: "Qwen/Qwen2-VL-7B-Instruct".to_string(),
                backend: Backend::Transformers,
                quantization: Quantization::None,
                device: "cuda:0".to_string(),
            },
            processing: ProcessingConfig {
                input_dir: PathBuf::from("./input"),
                output_dir: PathBuf::from("./output"),
                archive_dir: PathBuf::from("./archive"),
                archive_processed: true,
                image_dpi: 150,
                image_format: ImageFormat::Png,
            },
            output: OutputConfig {
                format: OutputFormat::Json,
                include_raw_text: false,
                confidence_threshold: 0.7,
            },
            inference: InferenceConfig {
                max_tokens: 4096,
                temperature: 0.1,
                server_url: "http://127.0.0.1:8765".to_string(),
            },
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// Load configuration from default location, or use defaults if not found
    pub fn load_or_default() -> Self {
        let default_path = Path::new("config/default.toml");
        if default_path.exists() {
            Self::load(default_path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load config: {}, using defaults", e);
                Self::default()
            })
        } else {
            Self::default()
        }
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))
    }
}
