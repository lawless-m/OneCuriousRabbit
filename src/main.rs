//! Invoice OCR - Local invoice extraction using multimodal LLMs

mod config;
mod inference;
mod output;
mod pdf;
mod types;
mod validation;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use config::{Config, OutputFormat};
use inference::InferenceClient;
use types::{InvoiceExtraction, ProcessingSummary};
use validation::{calculate_confidence, validate_extraction, Severity};

/// Maximum number of retries for failed extractions
const MAX_RETRIES: u32 = 2;

/// Delay between retries in milliseconds
const RETRY_DELAY_MS: u64 = 1000;

#[derive(Parser)]
#[command(name = "invoice-ocr")]
#[command(about = "Extract structured data from PDF invoices using multimodal LLMs")]
#[command(version)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process invoice PDF(s)
    Process {
        /// Path to PDF file or directory (or use --all for input directory)
        path: Option<PathBuf>,

        /// Process all PDFs in the input directory
        #[arg(long)]
        all: bool,

        /// Output format (json, csv, both)
        #[arg(short, long)]
        format: Option<OutputFormat>,

        /// Model to use for extraction
        #[arg(short, long)]
        model: Option<String>,

        /// Skip archiving processed files
        #[arg(long)]
        no_archive: bool,

        /// Number of retries for failed extractions
        #[arg(long, default_value = "2")]
        retries: u32,
    },

    /// Check system and server status
    Status,

    /// List available models
    Models,

    /// Initialize default configuration
    Init,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Load configuration
    let config = if let Some(config_path) = &cli.config {
        Config::load(config_path)?
    } else {
        Config::load_or_default()
    };

    match cli.command {
        Commands::Process { path, all, format, model: _, no_archive, retries } => {
            process_command(config, path, all, format, no_archive, retries)
        }
        Commands::Status => status_command(config),
        Commands::Models => models_command(),
        Commands::Init => init_command(),
    }
}

/// Process invoice(s)
fn process_command(
    mut config: Config,
    path: Option<PathBuf>,
    all: bool,
    format: Option<OutputFormat>,
    no_archive: bool,
    max_retries: u32,
) -> Result<()> {
    // Override format if specified
    if let Some(fmt) = format {
        config.output.format = fmt;
    }

    // Override archive setting if --no-archive specified
    if no_archive {
        config.processing.archive_processed = false;
    }

    // Collect PDFs to process
    let pdfs = if all {
        collect_pdfs_from_dir(&config.processing.input_dir)?
    } else if let Some(p) = path {
        if p.is_dir() {
            collect_pdfs_from_dir(&p)?
        } else {
            vec![p]
        }
    } else {
        anyhow::bail!("Provide a path or use --all to process input directory");
    };

    if pdfs.is_empty() {
        println!("No PDF files found to process.");
        return Ok(());
    }

    println!("Found {} PDF(s) to process", pdfs.len());

    // Create inference client
    let client = InferenceClient::new(&config.inference.server_url)
        .context("Failed to create inference client")?;

    // Check server status
    match client.check_status() {
        Ok(status) => {
            if !status.model_loaded {
                println!("Warning: Model not loaded on server. First request may be slow.");
            }
        }
        Err(e) => {
            anyhow::bail!(
                "Cannot connect to inference server at {}: {}\n\
                 Start the server with: python python/inference_server.py",
                config.inference.server_url,
                e
            );
        }
    }

    // Initialize processing summary
    let mut summary = ProcessingSummary::new();
    summary.total_files = pdfs.len();

    // Process each PDF
    let mut extractions: Vec<InvoiceExtraction> = Vec::new();

    for pdf_path in &pdfs {
        let filename = pdf_path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        println!("\nProcessing: {}", pdf_path.display());

        // Try processing with retries
        let result = process_with_retry(&client, &config, pdf_path, max_retries);

        match result {
            Ok(mut extraction) => {
                // Validate and calculate confidence
                let errors = validate_extraction(&extraction);
                extraction.confidence = calculate_confidence(&extraction, &errors);

                // Add validation errors as warnings
                for error in &errors {
                    if error.severity != Severity::Info {
                        extraction.warnings.push(format!("{}: {}", error.field, error.message));
                    }
                }

                // Write individual JSON output
                let output_name = pdf_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                let json_path = config.processing.output_dir.join(format!("{}.json", output_name));

                std::fs::create_dir_all(&config.processing.output_dir)?;
                output::write_json(&extraction, &json_path)?;

                // Report results
                println!("  Confidence: {:.0}% (header: {:.0}%, line items: {:.0}%)",
                    extraction.confidence.overall * 100.0,
                    extraction.confidence.header * 100.0,
                    extraction.confidence.line_items * 100.0
                );
                println!("  Extracted {} line items", extraction.line_items.len());

                if extraction.confidence.overall < config.output.confidence_threshold {
                    println!("  [LOW CONFIDENCE] - flagged for review");
                }

                if !extraction.warnings.is_empty() {
                    let error_count = errors.iter().filter(|e| e.severity == Severity::Error).count();
                    let warning_count = errors.iter().filter(|e| e.severity == Severity::Warning).count();
                    println!("  Validation: {} error(s), {} warning(s)", error_count, warning_count);
                }

                // Archive if configured
                if config.processing.archive_processed {
                    match archive_file(pdf_path, &config.processing.archive_dir) {
                        Ok(archive_path) => {
                            println!("  Archived to: {}", archive_path.display());
                        }
                        Err(e) => {
                            tracing::warn!("Failed to archive {}: {}", pdf_path.display(), e);
                        }
                    }
                }

                summary.record_success(&extraction, config.output.confidence_threshold);
                extractions.push(extraction);
            }
            Err(e) => {
                println!("  FAILED: {}", e);
                tracing::error!("Failed to process {}: {:?}", pdf_path.display(), e);
                summary.record_failure(filename, &e.to_string());
            }
        }
    }

    // Write batch outputs if needed
    if !extractions.is_empty() {
        match config.output.format {
            OutputFormat::Csv | OutputFormat::Both => {
                let csv_path = config.processing.output_dir.join("invoices.csv");
                output::write_headers_csv(&extractions, &csv_path)?;

                let items_path = config.processing.output_dir.join("line_items.csv");
                output::write_line_items_csv(&extractions, &items_path)?;
            }
            OutputFormat::Json => {}
        }

        if config.output.format == OutputFormat::Both {
            let excel_path = config.processing.output_dir.join("invoices.xlsx");
            output::write_excel(&extractions, &excel_path)?;
        }
    }

    // Print summary
    summary.print_summary();

    Ok(())
}

/// Process a PDF with retry logic
fn process_with_retry(
    client: &InferenceClient,
    config: &Config,
    pdf_path: &PathBuf,
    max_retries: u32,
) -> Result<InvoiceExtraction> {
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            println!("  Retry {}/{}...", attempt, max_retries);
            thread::sleep(Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
        }

        match process_single_pdf(client, config, pdf_path) {
            Ok(extraction) => return Ok(extraction),
            Err(e) => {
                tracing::warn!("Attempt {} failed: {}", attempt + 1, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error")))
}

/// Process a single PDF file
fn process_single_pdf(
    client: &InferenceClient,
    config: &Config,
    pdf_path: &PathBuf,
) -> Result<InvoiceExtraction> {
    // Validate PDF exists and is readable
    if !pdf_path.exists() {
        anyhow::bail!("File not found: {}", pdf_path.display());
    }

    // Check file size (basic corruption check)
    let metadata = std::fs::metadata(pdf_path)
        .with_context(|| format!("Cannot read file: {}", pdf_path.display()))?;

    if metadata.len() == 0 {
        anyhow::bail!("File is empty: {}", pdf_path.display());
    }

    // Create temp directory for images
    let temp_dir = pdf::create_temp_dir("invoice-ocr")?;

    // Convert PDF to images with error handling
    let images = match pdf::pdf_to_images(
        pdf_path,
        &temp_dir,
        config.processing.image_dpi,
        &config.processing.image_format,
    ) {
        Ok(imgs) => imgs,
        Err(e) => {
            // Cleanup on failure
            let _ = pdf::cleanup_temp_dir(&temp_dir);
            return Err(e.context("PDF conversion failed - file may be corrupt or password-protected"));
        }
    };

    println!("  Converted to {} page(s)", images.len());

    // Extract from first page (main extraction)
    let raw_extraction = client.extract(&images[0])
        .context("Failed to extract from first page")?;

    let mut extraction = InvoiceExtraction::from_raw(
        raw_extraction,
        pdf_path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
        config.model.name.clone(),
    );

    // Handle multi-page invoices
    if images.len() > 1 {
        for (i, image) in images.iter().enumerate().skip(1) {
            println!("  Processing page {}...", i + 1);

            let last_line = extraction.last_line_number();

            match client.extract_continuation(image, (i + 1) as u32, last_line) {
                Ok(continuation) => {
                    extraction.merge_continuation(continuation);
                }
                Err(e) => {
                    tracing::warn!("Failed to extract page {}: {}", i + 1, e);
                    extraction.warnings.push(format!("Page {} extraction failed: {}", i + 1, e));
                }
            }
        }
    }

    // Cleanup temp directory
    pdf::cleanup_temp_dir(&temp_dir)?;

    Ok(extraction)
}

/// Archive a file to the archive directory
fn archive_file(source: &PathBuf, archive_dir: &PathBuf) -> Result<PathBuf> {
    std::fs::create_dir_all(archive_dir)?;

    let filename = source.file_name()
        .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;

    let mut archive_path = archive_dir.join(filename);

    // Handle duplicate filenames
    if archive_path.exists() {
        let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("file");
        let ext = source.extension().and_then(|s| s.to_str()).unwrap_or("pdf");
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        archive_path = archive_dir.join(format!("{}_{}.{}", stem, timestamp, ext));
    }

    std::fs::rename(source, &archive_path)
        .with_context(|| format!("Failed to move {} to {}", source.display(), archive_path.display()))?;

    Ok(archive_path)
}

/// Collect all PDF files from a directory
fn collect_pdfs_from_dir(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        anyhow::bail!("Directory not found: {}", dir.display());
    }

    let pdfs: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                .unwrap_or(false)
        })
        .collect();

    Ok(pdfs)
}

/// Check status of inference server
fn status_command(config: Config) -> Result<()> {
    println!("Invoice OCR Status\n");

    // Check inference server
    println!("Inference Server: {}", config.inference.server_url);

    let client = InferenceClient::new(&config.inference.server_url)?;

    match client.check_status() {
        Ok(status) => {
            println!("  Status: {}", status.status);
            println!("  Model loaded: {}", status.model_loaded);
            if let Some(name) = status.model_name {
                println!("  Model: {}", name);
            }
            if let Some(device) = status.device {
                println!("  Device: {}", device);
            }
        }
        Err(e) => {
            println!("  Cannot connect: {}", e);
            println!("\n  Start the server with:");
            println!("    python python/inference_server.py");
        }
    }

    // Check directories
    println!("\nDirectories:");
    println!("  Input:   {} {}",
        config.processing.input_dir.display(),
        if config.processing.input_dir.exists() { "(exists)" } else { "(not found)" }
    );
    println!("  Output:  {} {}",
        config.processing.output_dir.display(),
        if config.processing.output_dir.exists() { "(exists)" } else { "(will be created)" }
    );
    println!("  Archive: {} {}",
        config.processing.archive_dir.display(),
        if config.processing.archive_dir.exists() { "(exists)" } else { "(will be created)" }
    );

    // Check poppler
    print!("\nPoppler (pdftoppm): ");
    match std::process::Command::new("pdftoppm").arg("-v").output() {
        Ok(_) => println!("installed"),
        Err(_) => println!("not found (install with: apt install poppler-utils)"),
    }

    Ok(())
}

/// List available models
fn models_command() -> Result<()> {
    println!("Available Models\n");
    println!("Primary:");
    println!("  Qwen/Qwen2-VL-7B-Instruct  - Recommended, ~16GB VRAM");
    println!("\nAlternatives:");
    println!("  Qwen/Qwen2-VL-2B-Instruct  - Faster, ~6GB VRAM");
    println!("  microsoft/Florence-2-large - Lightweight, ~3GB VRAM");
    println!("  vikhyatk/moondream2        - Very light, ~4GB VRAM");
    println!("\nConfigure in config/default.toml or use --model flag");

    Ok(())
}

/// Initialize default configuration
fn init_command() -> Result<()> {
    let config = Config::default();
    let config_path = PathBuf::from("config/default.toml");

    if config_path.exists() {
        println!("Configuration already exists at: {}", config_path.display());
        println!("Delete it first if you want to regenerate.");
        return Ok(());
    }

    config.save(&config_path)?;
    println!("Created default configuration at: {}", config_path.display());

    // Create directories
    std::fs::create_dir_all(&config.processing.input_dir)?;
    std::fs::create_dir_all(&config.processing.output_dir)?;
    std::fs::create_dir_all(&config.processing.archive_dir)?;
    println!("Created directories: input/, output/, archive/");

    Ok(())
}
