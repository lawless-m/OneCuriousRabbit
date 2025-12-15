#!/usr/bin/env python3
"""Create a simple test invoice PDF"""

from PIL import Image, ImageDraw, ImageFont
import subprocess

# Create image
width, height = 595, 842  # A4 at 72 DPI
img = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(img)

# Draw invoice content
y = 50
draw.text((50, y), "INVOICE", fill='black'); y += 40
draw.text((50, y), "ABC Company Ltd", fill='black'); y += 25
draw.text((50, y), "Invoice #: INV-2024-001", fill='black'); y += 25
draw.text((50, y), "Date: 2024-12-15", fill='black'); y += 25
draw.text((50, y), "Currency: GBP", fill='black'); y += 50

draw.text((50, y), "Item Description                 Qty    Price    Total", fill='black'); y += 30
draw.text((50, y), "Widget Premium Model             10     50.00    500.00", fill='black'); y += 25
draw.text((50, y), "Service Fee                       1    100.00    100.00", fill='black'); y += 50

draw.text((50, y), "Subtotal:  600.00 GBP", fill='black'); y += 25
draw.text((50, y), "VAT (20%): 120.00 GBP", fill='black'); y += 25
draw.text((50, y), "Total:     720.00 GBP", fill='black'); y += 25

# Save as PDF
img.save('input/test_invoice.pdf', 'PDF', resolution=72.0)
print("✓ Test invoice created: input/test_invoice.pdf")
