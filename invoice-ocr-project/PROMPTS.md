# Invoice Extraction Prompts

## Primary Extraction Prompt

This is the main prompt used to extract structured data from invoice images.

```
You are an invoice data extraction system. Extract all relevant information from this invoice image and return it as valid JSON.

Extract the following information:

HEADER FIELDS:
- supplier_name: The company issuing the invoice
- invoice_number: The unique invoice identifier
- invoice_date: Date the invoice was issued (format: YYYY-MM-DD)
- po_reference: Purchase order number if present (null if not found)
- currency: Three-letter currency code (e.g., GBP, EUR, USD)
- net_total: Total before tax
- vat_amount: Total VAT/tax amount
- gross_total: Total including tax
- payment_due_date: Payment due date if shown (format: YYYY-MM-DD, null if not found)

LINE ITEMS (for each item):
- line_number: Sequential line number (1, 2, 3...)
- product_code: Product/item code if shown (null if not present)
- description: Item description
- quantity: Number of items
- unit: Unit of measure (e.g., "each", "box", "kg", null if not specified)
- unit_price: Price per unit
- vat_rate: VAT percentage for this line if shown (null if not per-line)
- line_total: Total for this line

RULES:
1. Return ONLY valid JSON, no other text
2. Use null for fields that are not present or cannot be determined
3. Numbers should be numeric types, not strings
4. Dates must be in YYYY-MM-DD format
5. If you cannot read a value clearly, use null rather than guessing
6. Extract ALL line items, even if there are many

Return the data in this exact structure:
{
  "header": {
    "supplier_name": "",
    "invoice_number": "",
    "invoice_date": "",
    "po_reference": null,
    "currency": "",
    "net_total": 0.00,
    "vat_amount": 0.00,
    "gross_total": 0.00,
    "payment_due_date": null
  },
  "line_items": [
    {
      "line_number": 1,
      "product_code": null,
      "description": "",
      "quantity": 0,
      "unit": null,
      "unit_price": 0.00,
      "vat_rate": null,
      "line_total": 0.00
    }
  ]
}
```

## Multi-Page Prompt

For subsequent pages of a multi-page invoice:

```
This is page {page_number} of a multi-page invoice. Continue extracting line items.

If this page contains:
- Additional line items: Extract them with line_number continuing from {last_line_number}
- Summary/totals only: Return {"continuation": false, "additional_items": []}
- Mix of both: Extract items and note if totals are visible

Return JSON in this format:
{
  "continuation": true,
  "additional_items": [
    {
      "line_number": {next_line_number},
      "product_code": null,
      "description": "",
      "quantity": 0,
      "unit": null,
      "unit_price": 0.00,
      "vat_rate": null,
      "line_total": 0.00
    }
  ],
  "page_contains_totals": false
}
```

## Validation Prompt

Used to verify extraction and catch errors:

```
Review this extracted invoice data for errors or inconsistencies:

{extracted_json}

Check:
1. Do line item totals sum to the net total (within rounding tolerance)?
2. Does net + VAT = gross?
3. Are all required fields present?
4. Do any values seem implausible (negative quantities, dates in future, etc.)?

Return JSON:
{
  "valid": true/false,
  "issues": ["list of issues found"],
  "suggested_corrections": {}
}
```

## Supplier-Specific Prompt Additions

These can be appended to the main prompt for known suppliers with quirks:

### Template: Supplier with non-standard date format
```
NOTE: This supplier uses DD/MM/YYYY date format. Convert to YYYY-MM-DD in output.
```

### Template: Supplier with embedded PO in description
```
NOTE: This supplier embeds PO references in line item descriptions. Extract PO number from description if found in format "PO: XXXXX" or "Ref: XXXXX".
```

### Template: Supplier with VAT-inclusive pricing
```
NOTE: This supplier shows VAT-inclusive prices. The unit_price shown includes VAT. Calculate net values by dividing by 1.{vat_rate}.
```

## Confidence Assessment Prompt

For generating confidence scores:

```
Rate your confidence in each extracted field on a scale of 0.0 to 1.0:
- 1.0: Clearly visible and unambiguous
- 0.7-0.9: Visible but slightly unclear or partially obscured
- 0.4-0.6: Required interpretation or inference
- 0.0-0.3: Guessed or very uncertain

Return confidence scores for:
{
  "header_confidence": {
    "supplier_name": 0.0,
    "invoice_number": 0.0,
    "invoice_date": 0.0,
    "po_reference": 0.0,
    "net_total": 0.0,
    "vat_amount": 0.0,
    "gross_total": 0.0
  },
  "line_items_average_confidence": 0.0,
  "overall_confidence": 0.0
}
```

## Prompt Engineering Notes

### What works well:
- Explicit JSON structure examples
- Clear field definitions
- Handling of null/missing values
- Specific date format requirements

### What to avoid:
- Ambiguous field names
- Asking for "any other relevant information"
- Complex nested conditions
- Asking model to explain its reasoning (wastes tokens)

### Tuning for accuracy:
- Temperature: 0.1 or lower for structured extraction
- If model hallucinates fields, add "Use null if not clearly visible"
- If model truncates line items, increase max_tokens and add "Extract ALL items"
- If JSON is malformed, add "Return ONLY valid JSON, no markdown formatting"

### Known issues:
- Very long invoices (50+ line items) may need chunking
- Handwritten annotations can confuse extraction
- Multi-column layouts may scramble line item order
- Some models add markdown code blocks around JSON - strip these in post-processing
