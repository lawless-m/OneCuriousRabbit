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
                overall: 0.0,  // To be calculated
                header: 0.0,
                line_items: 0.0,
            },
            header: raw.header,
            line_items: raw.line_items,
            warnings: Vec::new(),
            raw_text: None,
        }
    }

    /// Validate the extraction and populate warnings
    pub fn validate(&mut self) {
        // Check if line items sum to net total
        let line_sum: f64 = self.line_items.iter().map(|l| l.line_total).sum();
        let tolerance = 0.01;

        if (line_sum - self.header.net_total).abs() > tolerance {
            self.warnings.push(format!(
                "Line items sum ({:.2}) does not match net total ({:.2})",
                line_sum, self.header.net_total
            ));
        }

        // Check if net + VAT = gross
        let calculated_gross = self.header.net_total + self.header.vat_amount;
        if (calculated_gross - self.header.gross_total).abs() > tolerance {
            self.warnings.push(format!(
                "Net ({:.2}) + VAT ({:.2}) = {:.2} does not match gross total ({:.2})",
                self.header.net_total, self.header.vat_amount,
                calculated_gross, self.header.gross_total
            ));
        }

        // Check for missing required fields
        if self.header.supplier_name.is_empty() {
            self.warnings.push("Missing supplier name".to_string());
        }
        if self.header.invoice_number.is_empty() {
            self.warnings.push("Missing invoice number".to_string());
        }
        if self.header.invoice_date.is_empty() {
            self.warnings.push("Missing invoice date".to_string());
        }

        // Calculate simple confidence based on warnings
        self.confidence.header = if self.warnings.is_empty() { 0.95 } else { 0.7 };
        self.confidence.line_items = if self.line_items.is_empty() { 0.0 } else { 0.9 };
        self.confidence.overall = (self.confidence.header + self.confidence.line_items) / 2.0;
    }
}
