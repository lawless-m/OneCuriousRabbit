//! HTTP bridge to Python inference server

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

use crate::types::RawExtraction;

/// Request to the inference server
#[derive(Debug, Serialize)]
struct InferenceRequest {
    image_path: String,
    prompt: String,
}

/// Response from the inference server
#[derive(Debug, Deserialize)]
struct InferenceResponse {
    success: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
}

/// Status response from the server
#[derive(Debug, Deserialize)]
pub struct ServerStatus {
    pub status: String,
    pub model_loaded: bool,
    pub model_name: Option<String>,
    pub device: Option<String>,
}

/// Client for communicating with the Python inference server
pub struct InferenceClient {
    client: reqwest::blocking::Client,
    base_url: String,
    prompt_template: String,
}

impl InferenceClient {
    /// Create a new inference client
    pub fn new(server_url: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120)) // Long timeout for inference
            .build()
            .context("Failed to create HTTP client")?;

        // Load prompt template
        let prompt_template = Self::load_prompt_template()?;

        Ok(Self {
            client,
            base_url: server_url.to_string(),
            prompt_template,
        })
    }

    /// Load the extraction prompt template from file
    fn load_prompt_template() -> Result<String> {
        let prompt_path = Path::new("prompts/invoice_extract.txt");
        if prompt_path.exists() {
            std::fs::read_to_string(prompt_path)
                .context("Failed to read prompt template")
        } else {
            // Use embedded default prompt
            Ok(include_str!("../../prompts/invoice_extract.txt").to_string())
        }
    }

    /// Check if the inference server is running and ready
    pub fn check_status(&self) -> Result<ServerStatus> {
        let url = format!("{}/status", self.base_url);

        let response = self.client
            .get(&url)
            .send()
            .context("Failed to connect to inference server")?;

        if !response.status().is_success() {
            bail!("Inference server returned error: {}", response.status());
        }

        response
            .json()
            .context("Failed to parse server status response")
    }

    /// Extract invoice data from an image
    pub fn extract(&self, image_path: &Path) -> Result<RawExtraction> {
        let url = format!("{}/extract", self.base_url);

        let request = InferenceRequest {
            image_path: image_path.to_string_lossy().to_string(),
            prompt: self.prompt_template.clone(),
        };

        tracing::info!("Sending extraction request for: {}", image_path.display());

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send extraction request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            bail!("Extraction request failed: {} - {}", status, body);
        }

        let inference_response: InferenceResponse = response
            .json()
            .context("Failed to parse extraction response")?;

        if !inference_response.success {
            bail!(
                "Extraction failed: {}",
                inference_response.error.unwrap_or_else(|| "Unknown error".to_string())
            );
        }

        let data = inference_response
            .data
            .ok_or_else(|| anyhow::anyhow!("No data in successful response"))?;

        serde_json::from_value(data)
            .context("Failed to parse extraction data into RawExtraction")
    }

    /// Extract from a continuation page (multi-page invoice)
    pub fn extract_continuation(
        &self,
        image_path: &Path,
        page_number: u32,
        last_line_number: u32,
    ) -> Result<crate::types::ContinuationExtraction> {
        let url = format!("{}/extract", self.base_url);

        let prompt = format!(
            r#"This is page {} of a multi-page invoice. Continue extracting line items.

If this page contains:
- Additional line items: Extract them with line_number continuing from {}
- Summary/totals only: Return {{"continuation": false, "additional_items": []}}
- Mix of both: Extract items and note if totals are visible

Return JSON in this format:
{{
  "continuation": true,
  "additional_items": [
    {{
      "line_number": {},
      "product_code": null,
      "description": "",
      "quantity": 0,
      "unit": null,
      "unit_price": 0.00,
      "vat_rate": null,
      "line_total": 0.00
    }}
  ],
  "page_contains_totals": false
}}"#,
            page_number,
            last_line_number,
            last_line_number + 1
        );

        let request = InferenceRequest {
            image_path: image_path.to_string_lossy().to_string(),
            prompt,
        };

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send continuation extraction request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            bail!("Continuation extraction failed: {} - {}", status, body);
        }

        let inference_response: InferenceResponse = response
            .json()
            .context("Failed to parse continuation response")?;

        if !inference_response.success {
            bail!(
                "Continuation extraction failed: {}",
                inference_response.error.unwrap_or_else(|| "Unknown error".to_string())
            );
        }

        let data = inference_response
            .data
            .ok_or_else(|| anyhow::anyhow!("No data in successful response"))?;

        serde_json::from_value(data)
            .context("Failed to parse continuation data")
    }
}
