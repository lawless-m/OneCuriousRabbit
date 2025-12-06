//! PDF to image conversion using Poppler's pdftoppm

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::ImageFormat;

/// Convert a PDF file to images (one per page)
pub fn pdf_to_images(
    pdf_path: &Path,
    output_dir: &Path,
    dpi: u32,
    format: &ImageFormat,
) -> Result<Vec<PathBuf>> {
    // Validate input
    if !pdf_path.exists() {
        bail!("PDF file not found: {}", pdf_path.display());
    }

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    // Generate output prefix
    let file_stem = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page");
    let output_prefix = output_dir.join(file_stem);

    // Determine format flag
    let format_flag = match format {
        ImageFormat::Png => "-png",
        ImageFormat::Jpeg => "-jpeg",
    };

    // Run pdftoppm
    let output = Command::new("pdftoppm")
        .arg(format_flag)
        .args(["-r", &dpi.to_string()])
        .arg(pdf_path)
        .arg(&output_prefix)
        .output()
        .context("Failed to execute pdftoppm. Is Poppler installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("pdftoppm failed: {}", stderr);
    }

    // Collect generated image files
    let extension = match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
    };

    let mut images: Vec<PathBuf> = std::fs::read_dir(output_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| ext == extension)
                .unwrap_or(false)
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(file_stem))
                    .unwrap_or(false)
        })
        .collect();

    // Sort by page number (filenames are like "invoice-1.png", "invoice-2.png")
    images.sort_by(|a, b| {
        let num_a = extract_page_number(a);
        let num_b = extract_page_number(b);
        num_a.cmp(&num_b)
    });

    if images.is_empty() {
        bail!("No images generated from PDF");
    }

    tracing::info!("Converted {} to {} page(s)", pdf_path.display(), images.len());

    Ok(images)
}

/// Extract page number from filename (e.g., "invoice-1.png" -> 1)
fn extract_page_number(path: &Path) -> u32 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit('-').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// Get the number of pages in a PDF without converting
pub fn get_page_count(pdf_path: &Path) -> Result<u32> {
    let output = Command::new("pdfinfo")
        .arg(pdf_path)
        .output()
        .context("Failed to execute pdfinfo. Is Poppler installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("pdfinfo failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.starts_with("Pages:") {
            if let Some(count) = line.split(':').nth(1) {
                return count.trim().parse().context("Failed to parse page count");
            }
        }
    }

    bail!("Could not determine page count from pdfinfo output")
}

/// Create a temporary directory for image conversion
pub fn create_temp_dir(prefix: &str) -> Result<PathBuf> {
    let temp_base = std::env::temp_dir();
    let temp_dir = temp_base.join(format!("{}-{}", prefix, std::process::id()));
    std::fs::create_dir_all(&temp_dir)
        .with_context(|| format!("Failed to create temp directory: {}", temp_dir.display()))?;
    Ok(temp_dir)
}

/// Clean up a temporary directory
pub fn cleanup_temp_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .with_context(|| format!("Failed to remove temp directory: {}", dir.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_page_number() {
        assert_eq!(extract_page_number(Path::new("invoice-1.png")), 1);
        assert_eq!(extract_page_number(Path::new("invoice-12.png")), 12);
        assert_eq!(extract_page_number(Path::new("my-doc-3.jpg")), 3);
    }
}
