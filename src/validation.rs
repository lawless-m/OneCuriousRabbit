//! Schema validation and confidence scoring for invoice extractions

use crate::types::{Confidence, InvoiceExtraction, InvoiceHeader, LineItem};

/// Validation errors found during schema validation
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Error,   // Must be fixed
    Warning, // Should review
    Info,    // FYI
}

/// Detailed confidence breakdown per field
#[derive(Debug, Clone, Default)]
pub struct DetailedConfidence {
    pub supplier_name: f64,
    pub invoice_number: f64,
    pub invoice_date: f64,
    pub po_reference: f64,
    pub currency: f64,
    pub totals: f64,
    pub line_items: f64,
}

impl DetailedConfidence {
    pub fn overall(&self) -> f64 {
        let fields = [
            self.supplier_name,
            self.invoice_number,
            self.invoice_date,
            self.currency,
            self.totals,
            self.line_items,
        ];
        fields.iter().sum::<f64>() / fields.len() as f64
    }
}

/// Validate an invoice extraction and return errors/warnings
pub fn validate_extraction(extraction: &InvoiceExtraction) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Validate header fields
    errors.extend(validate_header(&extraction.header));

    // Validate line items
    errors.extend(validate_line_items(&extraction.line_items));

    // Cross-validate totals
    errors.extend(validate_totals(&extraction.header, &extraction.line_items));

    errors
}

/// Validate header fields
fn validate_header(header: &InvoiceHeader) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Required fields
    if header.supplier_name.is_empty() {
        errors.push(ValidationError {
            field: "header.supplier_name".to_string(),
            message: "Supplier name is required".to_string(),
            severity: Severity::Error,
        });
    }

    if header.invoice_number.is_empty() {
        errors.push(ValidationError {
            field: "header.invoice_number".to_string(),
            message: "Invoice number is required".to_string(),
            severity: Severity::Error,
        });
    }

    if header.invoice_date.is_empty() {
        errors.push(ValidationError {
            field: "header.invoice_date".to_string(),
            message: "Invoice date is required".to_string(),
            severity: Severity::Error,
        });
    } else if !is_valid_date(&header.invoice_date) {
        errors.push(ValidationError {
            field: "header.invoice_date".to_string(),
            message: format!("Invalid date format: {} (expected YYYY-MM-DD)", header.invoice_date),
            severity: Severity::Warning,
        });
    }

    if header.currency.is_empty() {
        errors.push(ValidationError {
            field: "header.currency".to_string(),
            message: "Currency is required".to_string(),
            severity: Severity::Warning,
        });
    } else if header.currency.len() != 3 {
        errors.push(ValidationError {
            field: "header.currency".to_string(),
            message: format!("Currency should be 3-letter code: {}", header.currency),
            severity: Severity::Warning,
        });
    }

    // Validate totals are non-negative
    if header.net_total < 0.0 {
        errors.push(ValidationError {
            field: "header.net_total".to_string(),
            message: format!("Net total is negative: {:.2}", header.net_total),
            severity: Severity::Error,
        });
    }

    if header.vat_amount < 0.0 {
        errors.push(ValidationError {
            field: "header.vat_amount".to_string(),
            message: format!("VAT amount is negative: {:.2}", header.vat_amount),
            severity: Severity::Warning,
        });
    }

    if header.gross_total < 0.0 {
        errors.push(ValidationError {
            field: "header.gross_total".to_string(),
            message: format!("Gross total is negative: {:.2}", header.gross_total),
            severity: Severity::Error,
        });
    }

    // Validate payment due date if present
    if let Some(ref due_date) = header.payment_due_date {
        if !is_valid_date(due_date) {
            errors.push(ValidationError {
                field: "header.payment_due_date".to_string(),
                message: format!("Invalid due date format: {}", due_date),
                severity: Severity::Warning,
            });
        }
    }

    errors
}

/// Validate line items
fn validate_line_items(items: &[LineItem]) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if items.is_empty() {
        errors.push(ValidationError {
            field: "line_items".to_string(),
            message: "No line items found".to_string(),
            severity: Severity::Warning,
        });
        return errors;
    }

    for (i, item) in items.iter().enumerate() {
        let prefix = format!("line_items[{}]", i);

        if item.description.is_empty() {
            errors.push(ValidationError {
                field: format!("{}.description", prefix),
                message: format!("Line {} has empty description", item.line_number),
                severity: Severity::Warning,
            });
        }

        if item.quantity <= 0.0 {
            errors.push(ValidationError {
                field: format!("{}.quantity", prefix),
                message: format!("Line {} has invalid quantity: {}", item.line_number, item.quantity),
                severity: Severity::Warning,
            });
        }

        if item.unit_price < 0.0 {
            errors.push(ValidationError {
                field: format!("{}.unit_price", prefix),
                message: format!("Line {} has negative unit price: {:.2}", item.line_number, item.unit_price),
                severity: Severity::Warning,
            });
        }

        // Check line total calculation
        let expected_total = item.quantity * item.unit_price;
        if (expected_total - item.line_total).abs() > 0.02 {
            errors.push(ValidationError {
                field: format!("{}.line_total", prefix),
                message: format!(
                    "Line {} total mismatch: {} x {:.2} = {:.2}, but got {:.2}",
                    item.line_number, item.quantity, item.unit_price, expected_total, item.line_total
                ),
                severity: Severity::Info,
            });
        }

        // Check VAT rate is reasonable
        if let Some(vat_rate) = item.vat_rate {
            if !(0.0..=100.0).contains(&vat_rate) {
                errors.push(ValidationError {
                    field: format!("{}.vat_rate", prefix),
                    message: format!("Line {} has unusual VAT rate: {}%", item.line_number, vat_rate),
                    severity: Severity::Warning,
                });
            }
        }
    }

    // Check for duplicate line numbers
    let mut line_numbers: Vec<u32> = items.iter().map(|i| i.line_number).collect();
    line_numbers.sort();
    for window in line_numbers.windows(2) {
        if window[0] == window[1] {
            errors.push(ValidationError {
                field: "line_items".to_string(),
                message: format!("Duplicate line number: {}", window[0]),
                severity: Severity::Warning,
            });
        }
    }

    errors
}

/// Cross-validate totals
fn validate_totals(header: &InvoiceHeader, items: &[LineItem]) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let tolerance = 0.02; // 2 pence/cents tolerance for rounding

    // Check net + VAT = gross
    let calculated_gross = header.net_total + header.vat_amount;
    if (calculated_gross - header.gross_total).abs() > tolerance {
        errors.push(ValidationError {
            field: "header.gross_total".to_string(),
            message: format!(
                "Net ({:.2}) + VAT ({:.2}) = {:.2}, but gross is {:.2}",
                header.net_total, header.vat_amount, calculated_gross, header.gross_total
            ),
            severity: Severity::Warning,
        });
    }

    // Check line items sum to net total
    if !items.is_empty() {
        let line_sum: f64 = items.iter().map(|i| i.line_total).sum();
        if (line_sum - header.net_total).abs() > tolerance * items.len() as f64 {
            errors.push(ValidationError {
                field: "header.net_total".to_string(),
                message: format!(
                    "Line items sum ({:.2}) does not match net total ({:.2})",
                    line_sum, header.net_total
                ),
                severity: Severity::Warning,
            });
        }
    }

    errors
}

/// Calculate detailed confidence scores for an extraction
pub fn calculate_confidence(extraction: &InvoiceExtraction, errors: &[ValidationError]) -> Confidence {
    let mut detailed = DetailedConfidence::default();

    // Start with high confidence, reduce based on issues
    detailed.supplier_name = if extraction.header.supplier_name.is_empty() { 0.0 } else { 0.95 };
    detailed.invoice_number = if extraction.header.invoice_number.is_empty() { 0.0 } else { 0.95 };
    detailed.invoice_date = if extraction.header.invoice_date.is_empty() {
        0.0
    } else if is_valid_date(&extraction.header.invoice_date) {
        0.95
    } else {
        0.5
    };
    detailed.po_reference = 0.9; // Optional field
    detailed.currency = if extraction.header.currency.len() == 3 { 0.95 } else { 0.6 };

    // Totals confidence based on validation
    detailed.totals = 0.95;
    for error in errors {
        if error.field.contains("total") || error.field.contains("vat") {
            match error.severity {
                Severity::Error => detailed.totals -= 0.3,
                Severity::Warning => detailed.totals -= 0.15,
                Severity::Info => detailed.totals -= 0.05,
            }
        }
    }
    detailed.totals = detailed.totals.max(0.0);

    // Line items confidence
    if extraction.line_items.is_empty() {
        detailed.line_items = 0.0;
    } else {
        detailed.line_items = 0.9;
        let line_errors = errors.iter().filter(|e| e.field.contains("line_items")).count();
        detailed.line_items -= 0.1 * (line_errors as f64 / extraction.line_items.len() as f64);
        detailed.line_items = detailed.line_items.max(0.0);
    }

    // Calculate aggregate scores
    let header_confidence = (detailed.supplier_name
        + detailed.invoice_number
        + detailed.invoice_date
        + detailed.currency
        + detailed.totals)
        / 5.0;

    Confidence {
        overall: detailed.overall(),
        header: header_confidence,
        line_items: detailed.line_items,
    }
}

/// Check if a date string is in YYYY-MM-DD format
fn is_valid_date(date: &str) -> bool {
    if date.len() != 10 {
        return false;
    }

    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return false;
    }

    let year: Result<u32, _> = parts[0].parse();
    let month: Result<u32, _> = parts[1].parse();
    let day: Result<u32, _> = parts[2].parse();

    match (year, month, day) {
        (Ok(y), Ok(m), Ok(d)) => {
            (1900..=2100).contains(&y) && (1..=12).contains(&m) && (1..=31).contains(&d)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_date() {
        assert!(is_valid_date("2024-01-15"));
        assert!(is_valid_date("2023-12-31"));
        assert!(!is_valid_date("15-01-2024"));
        assert!(!is_valid_date("2024/01/15"));
        assert!(!is_valid_date("invalid"));
    }
}
