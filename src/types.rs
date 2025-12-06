//! Data types for invoice extraction

use serde::{Deserialize, Serialize};

/// Complete extraction result for an invoice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceExtraction {
    pub extraction_version: String,
    pub source_file: String,
    pub extracted_at: String,
    pub model_used: String,
    pub confidence: Confidence,
    pub header: InvoiceHeader,
    pub line_items: Vec<LineItem>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub raw_text: Option<String>,
}

/// Confidence scores for the extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confidence {
    pub overall: f64,
    pub header: f64,
    pub line_items: f64,
}

/// Invoice header fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceHeader {
    pub supplier_name: String,
    pub invoice_number: String,
    pub invoice_date: String,
    pub po_reference: Option<String>,
    pub currency: String,
    pub net_total: f64,
    pub vat_amount: f64,
    pub gross_total: f64,
    pub payment_due_date: Option<String>,
}

/// Individual line item from the invoice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    pub line_number: u32,
    pub product_code: Option<String>,
    pub description: String,
    pub quantity: f64,
    pub unit: Option<String>,
    pub unit_price: f64,
    pub vat_rate: Option<f64>,
    pub line_total: f64,
}

/// Raw extraction response from the model (before wrapping with metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawExtraction {
    pub header: InvoiceHeader,
    pub line_items: Vec<LineItem>,
}

/// Response from multi-page continuation extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationExtraction {
    pub continuation: bool,
    pub additional_items: Vec<LineItem>,
    pub page_contains_totals: bool,
}

/// Validation result from verification prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub suggested_corrections: Option<serde_json::Value>,
}

impl InvoiceExtraction {
    /// Create a new extraction result from raw model output
    pub fn from_raw(
        raw: RawExtraction,
        source_file: String,
        model_used: String,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();

        Self {
            extraction_version: "1.0".to_string(),
            source_file,
            extracted_at: now,
            model_used,
            confidence: Confidence {
                overall: 0.0,
                header: 0.0,
                line_items: 0.0,
            },
            header: raw.header,
            line_items: raw.line_items,
            warnings: Vec::new(),
            raw_text: None,
        }
    }

    /// Merge line items from a continuation page
    pub fn merge_continuation(&mut self, continuation: ContinuationExtraction) {
        if continuation.continuation {
            self.line_items.extend(continuation.additional_items);
        }
    }

    /// Get the last line number (for continuation pages)
    pub fn last_line_number(&self) -> u32 {
        self.line_items.last().map(|l| l.line_number).unwrap_or(0)
    }
}

/// Batch processing summary
#[derive(Debug, Clone, Default)]
pub struct ProcessingSummary {
    pub total_files: usize,
    pub successful: usize,
    pub failed: usize,
    pub low_confidence: usize,
    pub with_warnings: usize,
    pub total_line_items: usize,
    pub errors: Vec<(String, String)>, // (filename, error message)
}

impl ProcessingSummary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_success(&mut self, extraction: &InvoiceExtraction, threshold: f64) {
        self.successful += 1;
        self.total_line_items += extraction.line_items.len();

        if extraction.confidence.overall < threshold {
            self.low_confidence += 1;
        }
        if !extraction.warnings.is_empty() {
            self.with_warnings += 1;
        }
    }

    pub fn record_failure(&mut self, filename: &str, error: &str) {
        self.failed += 1;
        self.errors.push((filename.to_string(), error.to_string()));
    }

    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(50));
        println!("Processing Summary");
        println!("{}", "=".repeat(50));
        println!("Total files:      {}", self.total_files);
        println!("Successful:       {}", self.successful);
        println!("Failed:           {}", self.failed);
        println!("Low confidence:   {}", self.low_confidence);
        println!("With warnings:    {}", self.with_warnings);
        println!("Total line items: {}", self.total_line_items);

        if !self.errors.is_empty() {
            println!("\nFailed files:");
            for (filename, error) in &self.errors {
                println!("  {} - {}", filename, error);
            }
        }
    }
}
