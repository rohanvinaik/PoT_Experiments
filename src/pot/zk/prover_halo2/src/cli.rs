use clap::{Arg, ArgMatches, Command, ValueEnum};
use std::path::PathBuf;
use std::io::{self, Read, Write};
use std::fs::File;

/// Output format options
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Json,
    Binary,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Binary => write!(f, "binary"),
        }
    }
}

/// Common CLI arguments shared across binaries
pub struct CommonArgs {
    pub input: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub verbose: bool,
    pub params_k: Option<u32>,
    pub format: OutputFormat,
}

impl CommonArgs {
    pub fn from_matches(matches: &ArgMatches) -> Self {
        Self {
            input: matches.get_one::<PathBuf>("input").cloned(),
            output: matches.get_one::<PathBuf>("output").cloned(),
            verbose: matches.get_flag("verbose"),
            params_k: matches.get_one::<u32>("params-k").copied(),
            format: matches.get_one::<OutputFormat>("format").cloned().unwrap_or(OutputFormat::Json),
        }
    }
}

/// Build common CLI arguments that all binaries share
pub fn add_common_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("input")
            .short('i')
            .long("input")
            .value_name("FILE")
            .help("Input file (default: stdin)")
            .value_parser(clap::value_parser!(PathBuf))
    )
    .arg(
        Arg::new("output")
            .short('o')
            .long("output")
            .value_name("FILE")
            .help("Output file (default: stdout)")
            .value_parser(clap::value_parser!(PathBuf))
    )
    .arg(
        Arg::new("verbose")
            .short('v')
            .long("verbose")
            .action(clap::ArgAction::SetTrue)
            .help("Enable verbose output")
    )
    .arg(
        Arg::new("params-k")
            .long("params-k")
            .value_name("SIZE")
            .help("Circuit size parameter (log2 of constraint count)")
            .value_parser(clap::value_parser!(u32).range(8..=25))
    )
    .arg(
        Arg::new("format")
            .long("format")
            .value_name("FORMAT")
            .help("Output format")
            .value_parser(clap::value_parser!(OutputFormat))
            .default_value("json")
    )
}

/// Read input from file or stdin
pub fn read_input(input_path: Option<&PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    let mut content = String::new();
    
    match input_path {
        Some(path) => {
            if !path.exists() {
                return Err(format!("Input file does not exist: {}", path.display()).into());
            }
            let mut file = File::open(path)?;
            file.read_to_string(&mut content)?;
        }
        None => {
            io::stdin().read_to_string(&mut content)?;
        }
    }
    
    Ok(content)
}

/// Write output to file or stdout
pub fn write_output(
    output_path: Option<&PathBuf>,
    content: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match output_path {
        Some(path) => {
            // Create parent directories if they don't exist
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            
            let mut file = File::create(path)?;
            file.write_all(content.as_bytes())?;
            file.flush()?;
            
            if verbose {
                eprintln!("Output written to: {}", path.display());
            }
        }
        None => {
            print!("{}", content);
            io::stdout().flush()?;
        }
    }
    
    Ok(())
}

/// Print verbose message if verbose mode is enabled
pub fn verbose_println(verbose: bool, message: &str) {
    if verbose {
        eprintln!("[VERBOSE] {}", message);
    }
}

/// Print error message and exit with error code
pub fn error_exit(message: &str) -> ! {
    eprintln!("Error: {}", message);
    std::process::exit(1);
}

/// Print warning message
pub fn warn_println(message: &str) {
    eprintln!("Warning: {}", message);
}

/// Validate that input file exists (if provided)
pub fn validate_input_file(path: Option<&PathBuf>) -> Result<(), String> {
    if let Some(path) = path {
        if !path.exists() {
            return Err(format!("Input file does not exist: {}", path.display()));
        }
        if !path.is_file() {
            return Err(format!("Input path is not a file: {}", path.display()));
        }
    }
    Ok(())
}

/// Validate that output file directory exists or can be created
pub fn validate_output_file(path: Option<&PathBuf>) -> Result<(), String> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            if parent != std::path::Path::new("") && !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Cannot create output directory {}: {}", parent.display(), e))?;
            }
        }
        
        // Check if we can write to the file
        if path.exists() && path.metadata().map(|m| m.permissions().readonly()).unwrap_or(false) {
            return Err(format!("Output file is read-only: {}", path.display()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Binary.to_string(), "binary");
    }

    #[test]
    fn test_read_input_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_input.txt");
        let mut file = File::create(&file_path).unwrap();
        write!(file, "test content").unwrap();
        
        let content = read_input(Some(&file_path)).unwrap();
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_read_input_file_not_exists() {
        let path = PathBuf::from("/nonexistent/file.txt");
        let result = read_input(Some(&path));
        assert!(result.is_err());
    }

    #[test]
    fn test_write_output_to_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_output.txt");
        
        write_output(Some(&file_path), "test content", false).unwrap();
        
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_validate_input_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        File::create(&file_path).unwrap();
        
        assert!(validate_input_file(Some(&file_path)).is_ok());
        
        let nonexistent = dir.path().join("nonexistent.txt");
        assert!(validate_input_file(Some(&nonexistent)).is_err());
    }

    #[test]
    fn test_validate_output_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("subdir").join("test.txt");
        
        // Should create the parent directory
        assert!(validate_output_file(Some(&file_path)).is_ok());
        assert!(file_path.parent().unwrap().exists());
    }
}