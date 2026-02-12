# Figure dimensions for kaobook LaTeX class
# Based on the kaobook.cls geometry settings

# Key measurements from kaobook class (in mm):
# textwidth = 107mm (4.21 inches)
# marginparwidth = 49.4mm (1.94 inches) 
# marginparsep = 8.2mm (0.32 inches)
# Total page width with margins = 107 + 8.2 + 49.4 = 164.6mm (6.48 inches)

# Convert to inches for R (ggplot2 uses inches)
textwidth_inches <- 107 / 25.4  # 4.21 inches
marginparwidth_inches <- 49.4 / 25.4  # 1.94 inches
fullwidth_inches <- (107 + 8.2 + 49.4) / 25.4  # 6.48 inches

# Figure specifications
FIGURE_SPECS <- list(
  # Regular figures (fit in text column)
  regular = list(
    width = textwidth_inches,    # 4.21 inches
    height = textwidth_inches * 0.618,  # Golden ratio height (2.6 inches)
    dpi = 300,
    base_size = 9,  # Font size for axis labels, legends
    title_size = 10,
    caption_size = 8
  ),
  
  # Wide figures (span text + margin)
  wide = list(
    width = fullwidth_inches,    # 6.48 inches  
    height = fullwidth_inches * 0.5,  # Wider aspect ratio (3.24 inches)
    dpi = 300,
    base_size = 10,
    title_size = 11,
    caption_size = 8
  ),
  
  # Square figures (for heatmaps, etc.)
  square = list(
    width = textwidth_inches,    # 4.21 inches
    height = textwidth_inches,   # 4.21 inches (square)
    dpi = 300,
    base_size = 9,
    title_size = 10,
    caption_size = 8
  ),
  
  # Margin figures (fit in margin)
  margin = list(
    width = marginparwidth_inches,  # 1.94 inches
    height = marginparwidth_inches * 1.2,  # Slightly taller (2.33 inches)
    dpi = 300,
    base_size = 7,  # Smaller font for margin
    title_size = 8,
    caption_size = 6
  ),
  
  # Multi-panel figures (need more height)
  multipanel = list(
    width = fullwidth_inches,    # 6.48 inches
    height = fullwidth_inches * 0.75,  # Taller for multiple panels (4.86 inches)
    dpi = 300,
    base_size = 9,
    title_size = 10,
    caption_size = 8
  )
)

# Function to get figure specifications
get_figure_specs <- function(type = "regular") {
  if (!type %in% names(FIGURE_SPECS)) {
    warning("Unknown figure type '", type, "'. Using 'regular'.")
    type <- "regular"
  }
  return(FIGURE_SPECS[[type]])
}

# Function to create consistent theme for paper figures
paper_theme <- function(spec) {
  theme_bw(base_size = spec$base_size) +
    theme(
      # Text sizes
      plot.title = element_text(size = spec$title_size, face = "bold"),
      plot.subtitle = element_text(size = spec$base_size - 1),
      axis.title = element_text(size = spec$base_size),
      axis.text = element_text(size = spec$base_size - 1),
      legend.title = element_text(size = spec$base_size, face = "bold"),
      legend.text = element_text(size = spec$base_size - 1),
      strip.text = element_text(size = spec$base_size, face = "bold"),
      
      # Layout
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.margin = margin(t = 5, unit = "pt"),
      
      # Panel spacing
      panel.spacing = unit(0.5, "lines"),
      strip.background = element_rect(fill = "grey95", color = "black"),
      
      # Margins (tight for paper)
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5, unit = "pt")
    )
}

# Function to save figure with proper specifications
save_paper_figure <- function(plot, filename, type = "regular", formats = c("pdf", "png")) {
  spec <- get_figure_specs(type)
  
  # Create directory if it doesn't exist
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  
  # Save in requested formats
  for (format in formats) {
    full_filename <- paste0(tools::file_path_sans_ext(filename), ".", format)
    
    if (format == "pdf") {
      ggsave(full_filename, plot, 
             width = spec$width, height = spec$height, 
             units = "in", dpi = spec$dpi,
             device = "pdf", useDingbats = FALSE)
    } else if (format == "png") {
      ggsave(full_filename, plot, 
             width = spec$width, height = spec$height, 
             units = "in", dpi = spec$dpi,
             device = "png", type = "cairo")
    }
  }
  
  cat("Saved", type, "figure:", filename, "\n")
}

# Color scheme for consistent paper figures
PAPER_COLORS <- list(
  primary = "#2E86AB",      # Blue
  secondary = "#A23B72",    # Magenta  
  accent = "#F18F01",       # Orange
  neutral = "#C5C3C6",      # Gray
  success = "#46A649",      # Green
  warning = "#FD7F28",      # Orange-red
  
  # For null/overt subjects
  null = "#2E86AB",         # Blue for null
  overt = "#A23B72",        # Magenta for overt
  
  # For model comparisons (colorblind friendly)
  models = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"),
  
  # For linguistic forms (maintaining original colors)
  forms = c("both_negation" = "#E63946",
           "complex_emb" = "#F77F00", 
           "complex_long" = "#FCBF49",
           "context_negation" = "#2A9D8F",
           "default" = "#264653",
           "target_negation" = "#7209B7"),
  
  # Grayscale for supplementary figures
  gray_scale = c("#2b2b2b", "#555555", "#7f7f7f", "#a9a9a9", "#d3d3d3")
)

cat("Figure specifications loaded for kaobook LaTeX class:\n")
cat("- Text width:", textwidth_inches, "inches\n")
cat("- Full width:", fullwidth_inches, "inches\n") 
cat("- Margin width:", marginparwidth_inches, "inches\n")