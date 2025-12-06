//! Output formatting for JSON, CSV, and Excel

use anyhow::{Context, Result};
use std::path::Path;

use crate::types::InvoiceExtraction;

/// Write extraction result to JSON file
pub fn write_json(extraction: &InvoiceExtraction, path: &Path) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create JSON file: {}", path.display()))?;

    serde_json::to_writer_pretty(file, extraction)
        .with_context(|| format!("Failed to write JSON to: {}", path.display()))?;

    tracing::info!("Wrote JSON output to: {}", path.display());
    Ok(())
}

/// Write multiple extractions to a CSV file (headers only)
pub fn write_headers_csv(extractions: &[InvoiceExtraction], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("Failed to create CSV file: {}", path.display()))?;

    // Write header row
    writer.write_record([
        "source_file",
        "supplier_name",
        "invoice_number",
        "invoice_date",
        "po_reference",
        "currency",
        "net_total",
        "vat_amount",
        "gross_total",
        "payment_due_date",
        "confidence",
        "warnings",
    ])?;

    // Write data rows
    for extraction in extractions {
        writer.write_record([
            &extraction.source_file,
            &extraction.header.supplier_name,
            &extraction.header.invoice_number,
            &extraction.header.invoice_date,
            extraction.header.po_reference.as_deref().unwrap_or(""),
            &extraction.header.currency,
            &extraction.header.net_total.to_string(),
            &extraction.header.vat_amount.to_string(),
            &extraction.header.gross_total.to_string(),
            extraction.header.payment_due_date.as_deref().unwrap_or(""),
            &extraction.confidence.overall.to_string(),
            &extraction.warnings.join("; "),
        ])?;
    }

    writer.flush()?;
    tracing::info!("Wrote headers CSV to: {}", path.display());
    Ok(())
}

/// Write line items to a separate CSV file
pub fn write_line_items_csv(extractions: &[InvoiceExtraction], path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("Failed to create line items CSV: {}", path.display()))?;

    // Write header row
    writer.write_record([
        "source_file",
        "invoice_number",
        "line_number",
        "product_code",
        "description",
        "quantity",
        "unit",
        "unit_price",
        "vat_rate",
        "line_total",
    ])?;

    // Write data rows
    for extraction in extractions {
        for item in &extraction.line_items {
            writer.write_record([
                &extraction.source_file,
                &extraction.header.invoice_number,
                &item.line_number.to_string(),
                item.product_code.as_deref().unwrap_or(""),
                &item.description,
                &item.quantity.to_string(),
                item.unit.as_deref().unwrap_or(""),
                &item.unit_price.to_string(),
                &item.vat_rate.map(|v| v.to_string()).unwrap_or_default(),
                &item.line_total.to_string(),
            ])?;
        }
    }

    writer.flush()?;
    tracing::info!("Wrote line items CSV to: {}", path.display());
    Ok(())
}

/// Write extractions to an Excel file with two sheets
pub fn write_excel(extractions: &[InvoiceExtraction], path: &Path) -> Result<()> {
    use rust_xlsxwriter::{Format, Workbook};

    let mut workbook = Workbook::new();

    // Header format
    let header_format = Format::new().set_bold();

    // Create Headers sheet
    let headers_sheet = workbook.add_worksheet();
    headers_sheet.set_name("Invoices")?;

    // Write headers
    let header_cols = [
        "Source File",
        "Supplier",
        "Invoice Number",
        "Invoice Date",
        "PO Reference",
        "Currency",
        "Net Total",
        "VAT Amount",
        "Gross Total",
        "Due Date",
        "Confidence",
    ];

    for (col, header) in header_cols.iter().enumerate() {
        headers_sheet.write_string_with_format(0, col as u16, *header, &header_format)?;
    }

    // Write data
    for (row, extraction) in extractions.iter().enumerate() {
        let row = (row + 1) as u32;
        headers_sheet.write_string(row, 0, &extraction.source_file)?;
        headers_sheet.write_string(row, 1, &extraction.header.supplier_name)?;
        headers_sheet.write_string(row, 2, &extraction.header.invoice_number)?;
        headers_sheet.write_string(row, 3, &extraction.header.invoice_date)?;
        headers_sheet.write_string(row, 4, extraction.header.po_reference.as_deref().unwrap_or(""))?;
        headers_sheet.write_string(row, 5, &extraction.header.currency)?;
        headers_sheet.write_number(row, 6, extraction.header.net_total)?;
        headers_sheet.write_number(row, 7, extraction.header.vat_amount)?;
        headers_sheet.write_number(row, 8, extraction.header.gross_total)?;
        headers_sheet.write_string(row, 9, extraction.header.payment_due_date.as_deref().unwrap_or(""))?;
        headers_sheet.write_number(row, 10, extraction.confidence.overall)?;
    }

    // Create Line Items sheet
    let items_sheet = workbook.add_worksheet();
    items_sheet.set_name("Line Items")?;

    let item_cols = [
        "Source File",
        "Invoice Number",
        "Line",
        "Product Code",
        "Description",
        "Quantity",
        "Unit",
        "Unit Price",
        "VAT Rate",
        "Line Total",
    ];

    for (col, header) in item_cols.iter().enumerate() {
        items_sheet.write_string_with_format(0, col as u16, *header, &header_format)?;
    }

    let mut row: u32 = 1;
    for extraction in extractions {
        for item in &extraction.line_items {
            items_sheet.write_string(row, 0, &extraction.source_file)?;
            items_sheet.write_string(row, 1, &extraction.header.invoice_number)?;
            items_sheet.write_number(row, 2, item.line_number as f64)?;
            items_sheet.write_string(row, 3, item.product_code.as_deref().unwrap_or(""))?;
            items_sheet.write_string(row, 4, &item.description)?;
            items_sheet.write_number(row, 5, item.quantity)?;
            items_sheet.write_string(row, 6, item.unit.as_deref().unwrap_or(""))?;
            items_sheet.write_number(row, 7, item.unit_price)?;
            if let Some(vat) = item.vat_rate {
                items_sheet.write_number(row, 8, vat)?;
            }
            items_sheet.write_number(row, 9, item.line_total)?;
            row += 1;
        }
    }

    workbook.save(path)?;
    tracing::info!("Wrote Excel output to: {}", path.display());
    Ok(())
}
