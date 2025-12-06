# Supplier-Specific Prompt Templates

Place supplier-specific prompt templates in this directory. The filename (without extension)
will be used to match against the detected supplier name.

## How it works

1. First extraction uses the default prompt to identify the supplier
2. If a matching template exists, the invoice is re-extracted using the supplier-specific prompt
3. Supplier matching is case-insensitive and uses partial matching

## Naming convention

- Use lowercase, hyphenated filenames
- Example: `acme-corp.txt` matches "ACME Corp", "Acme Corporation", etc.

## Template structure

Templates should follow the same JSON structure as the default prompt.
Add supplier-specific instructions for:
- Known field locations
- Custom field mappings
- VAT handling quirks
- Date format preferences
