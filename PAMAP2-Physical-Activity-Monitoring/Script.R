##############################################
# Project: PAMAP2 Activity Recognition (Capstone Parallel Project)
# Author: Fernando Marcelo Parodi
#
##############################################

##############################################
# CONSOLE LOGGING UTILITIES (for step-by-step tracking)
##############################################
ts_now <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

log_step <- function(msg) cat(sprintf("\n[%s] === %s ===\n", ts_now(), msg))
log_info <- function(msg) cat(sprintf("[%s] INFO: %s\n", ts_now(), msg))
log_warn <- function(msg) cat(sprintf("[%s] WARN: %s\n", ts_now(), msg))

print_head <- function(df, n = 6, title = NULL) {
  if (!is.null(title)) log_info(title)
  print(utils::head(df, n))
}

log_table_preview <- function(df, n = 10, title = NULL) {
  if (!is.null(title)) log_info(title)
  if (nrow(df) == 0) {
    log_warn("Table is empty.")
  } else {
    print(df %>% dplyr::slice_head(n = n))
    log_info(sprintf("Rows: %s | Cols: %s", nrow(df), ncol(df)))
  }
}

##############################################
# STEP 01  DATA INGESTION
##############################################
log_step("STEP 01  DATA INGESTION (download + unzip + nested unzip + ingestion + audits)")

############################
# 0) Packages
############################
required_pkgs <- c(
  "callr", "curl",
  "data.table", "dplyr", "tidyr", "stringr", "purrr", "readr", "janitor",
  "ggplot2", "scales", "lubridate"
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing) > 0) install.packages(missing, dependencies = TRUE)
}
install_if_missing(required_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))
set.seed(42)

log_info("Packages loaded.")

############################
# 1) Resolve project_root and setwd(project_root)
############################
get_script_dir <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (nzchar(p) && file.exists(p)) return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", args, value = TRUE)
  if (length(hit) > 0) {
    p <- sub("^--file=", "", hit[1])
    if (nzchar(p) && file.exists(p)) return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  }
  NA_character_
}

project_root <- Sys.getenv("PAMAP2_PROJECT_ROOT", unset = NA_character_)
if (is.na(project_root) || !nzchar(project_root)) {
  script_dir <- get_script_dir()
  project_root <- if (!is.na(script_dir)) script_dir else getwd()
}
project_root <- normalizePath(project_root, winslash = "/", mustWork = FALSE)

setwd(project_root)
log_info(paste0("Working directory set to: ", normalizePath(getwd(), winslash = "/", mustWork = FALSE)))
log_info(paste0("Project root:            ", project_root))

############################
# 2) Paths
############################
dataset_original_dir <- file.path(project_root, "dataset_original")
dataset_extract_dir  <- file.path(dataset_original_dir, "extracted")

dir.create(dataset_original_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(dataset_extract_dir,  showWarnings = FALSE, recursive = TRUE)

out_processed_dir <- file.path(project_root, "data", "processed")
out_ingestion_dir <- file.path(project_root, "outputs", "ingestion")

dir.create(out_processed_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_ingestion_dir, showWarnings = FALSE, recursive = TRUE)

pamap2_zip_url  <- "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
pamap2_zip_path <- file.path(dataset_original_dir, "pamap2_physical_activity_monitoring.zip")
pamap2_zip_part <- paste0(pamap2_zip_path, ".part")

log_info(paste0("dataset_original_dir: ", dataset_original_dir))
log_info(paste0("dataset_extract_dir:  ", dataset_extract_dir))
log_info(paste0("pamap2_zip_path:      ", pamap2_zip_path))

############################
# 3) Helpers
############################
MB <- 1024^2
`%+%` <- function(a, b) paste0(a, b)
`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x

stop_with_context <- function(msg, err = NULL) {
  if (!is.null(err)) stop(paste0(msg, "\nUnderlying error: ", conditionMessage(err)), call. = FALSE)
  stop(msg, call. = FALSE)
}

is_writable_dir <- function(dir_path) {
  if (!dir.exists(dir_path)) return(FALSE)
  testfile <- file.path(dir_path, paste0(".__writetest__", Sys.getpid()))
  ok <- tryCatch({
    con <- file(testfile, open = "wb")
    writeBin(as.raw(1), con)
    close(con)
    file.remove(testfile)
    TRUE
  }, error = function(e) FALSE)
  ok
}

get_free_mb <- function(path_dir) {
  path_dir <- normalizePath(path_dir, winslash = "/", mustWork = FALSE)
  
  if (.Platform$OS.type == "unix") {
    out <- tryCatch(system2("df", c("-Pm", path_dir), stdout = TRUE, stderr = TRUE), error = function(e) NULL)
    if (is.null(out) || length(out) < 2) return(NA_real_)
    parts <- strsplit(trimws(out[length(out)]), "\\s+")[[1]]
    if (length(parts) >= 4) return(as.numeric(parts[4]))
    return(NA_real_)
  }
  
  if (.Platform$OS.type == "windows") {
    drive <- toupper(substr(path_dir, 1, 1))
    if (!grepl("^[A-Z]$", drive)) return(NA_real_)
    cmd <- sprintf("(Get-PSDrive -Name %s).Free/1MB", drive)
    out <- tryCatch(system2("powershell", c("-NoProfile", "-Command", cmd), stdout = TRUE, stderr = TRUE), error = function(e) NULL)
    if (is.null(out) || length(out) == 0) return(NA_real_)
    val <- suppressWarnings(as.numeric(trimws(out[1])))
    if (is.finite(val)) return(val)
    return(NA_real_)
  }
  
  NA_real_
}

parse_headers_robust <- function(hdrs) {
  if (is.null(hdrs)) return(setNames(character(0), character(0)))
  
  if (is.character(hdrs) && length(hdrs) > 0 && !is.null(names(hdrs)) && any(nzchar(names(hdrs)))) {
    return(hdrs)
  }
  
  parsed <- tryCatch(curl::parse_headers(hdrs), error = function(e) NULL)
  if (!is.null(parsed) && length(parsed) > 0) {
    v <- unlist(parsed, use.names = TRUE)
    v <- v[!is.na(v)]
    return(v)
  }
  
  if (is.character(hdrs) && length(hdrs) > 0) {
    txt <- paste(hdrs, collapse = "\n")
    lines <- unlist(strsplit(txt, "\r?\n"))
    lines <- trimws(lines)
    lines <- lines[grepl(":", lines)]
    if (length(lines) == 0) return(setNames(character(0), character(0)))
    
    keys <- sub(":.*$", "", lines)
    vals <- sub("^[^:]+:\\s*", "", lines)
    keys <- tolower(trimws(keys))
    out <- vals
    names(out) <- keys
    return(out)
  }
  
  setNames(character(0), character(0))
}

get_remote_content_length <- function(url) {
  h <- curl::new_handle()
  curl::handle_setopt(
    h,
    nobody = TRUE,
    header = TRUE,
    followlocation = TRUE,
    useragent = "R-curl (PAMAP2 Capstone)"
  )
  
  res <- tryCatch(curl::curl_fetch_memory(url, handle = h), error = function(e) NULL)
  if (is.null(res)) return(NA_real_)
  
  hdr_vec <- parse_headers_robust(res$headers)
  if (length(hdr_vec) == 0) return(NA_real_)
  
  nm <- tolower(names(hdr_vec))
  idx <- match("content-length", nm)
  if (is.na(idx)) return(NA_real_)
  
  cl <- hdr_vec[[idx]]
  cl_num <- suppressWarnings(as.numeric(cl))
  if (!is.finite(cl_num) || cl_num <= 0) return(NA_real_)
  cl_num
}

progress_file_size <- function(part_path) {
  tmp <- paste0(part_path, ".curltmp")
  if (file.exists(tmp)) return(file.info(tmp)$size %||% 0)
  if (file.exists(part_path)) return(file.info(part_path)$size %||% 0)
  0
}

############################
# 4) Download
############################
download_zip_resumable <- function(url, final_zip, part_zip, attempts = 3, min_ok_ratio = 0.99) {
  dest_dir <- dirname(final_zip)
  if (!dir.exists(dest_dir)) dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  if (!is_writable_dir(dest_dir)) stop_with_context(sprintf("Destination directory is not writable: %s", dest_dir))
  
  free_mb <- get_free_mb(dest_dir)
  if (is.finite(free_mb)) {
    log_info(sprintf("Free space detected: ~%.0f MB", free_mb))
    if (free_mb < 1200) stop_with_context(sprintf("Insufficient free space (~%.0f MB). Need at least ~1.2 GB free.", free_mb))
  } else {
    log_warn("Free space could not be detected on this system.")
  }
  
  expected_bytes <- get_remote_content_length(url)
  if (is.finite(expected_bytes)) {
    log_info(sprintf("Remote Content-Length: %.1f MB", expected_bytes / MB))
  } else {
    log_warn("Remote Content-Length not available. Will validate by unzip + folder detection.")
  }
  
  if (file.exists(final_zip) && file.info(final_zip)$size > 0 && is.finite(expected_bytes)) {
    local_final <- file.info(final_zip)$size
    if (local_final >= expected_bytes * min_ok_ratio) {
      log_info(sprintf("ZIP already complete: %s (%.1f MB). Skipping download.", final_zip, local_final / MB))
      return(invisible(TRUE))
    }
  }
  
  part_curltmp <- paste0(part_zip, ".curltmp")
  
  if (file.exists(part_curltmp) && !file.exists(part_zip)) {
    log_warn("Detected stale partial curl temp file. Normalizing to .part.")
    ok <- file.rename(part_curltmp, part_zip)
    if (!isTRUE(ok)) {
      ok2 <- file.copy(part_curltmp, part_zip, overwrite = TRUE)
      if (isTRUE(ok2)) file.remove(part_curltmp) else stop_with_context("Could not normalize stale .curltmp to .part.")
    }
  }
  
  log_info(paste0("Starting (or resuming) download into: ", part_zip))
  log_info(url)
  
  for (i in seq_len(attempts)) {
    offset <- if (file.exists(part_zip)) file.info(part_zip)$size else 0
    if (!is.finite(offset)) offset <- 0
    
    if (offset > 0) {
      log_info(sprintf("Attempt %d/%d: resuming from %.1f MB", i, attempts, offset / MB))
    } else {
      log_info(sprintf("Attempt %d/%d: starting fresh download", i, attempts))
      if (file.exists(part_zip)) file.remove(part_zip)
      if (file.exists(part_curltmp)) file.remove(part_curltmp)
      offset <- 0
    }
    
    p <- callr::r_bg(
      func = function(u, part_file, resume_offset) {
        h <- curl::new_handle()
        curl::handle_setopt(
          h,
          followlocation = TRUE,
          useragent = "R-curl (PAMAP2 Capstone)",
          connecttimeout = 30,
          low_speed_limit = 1,
          low_speed_time  = 60
        )
        
        if (!is.null(resume_offset) && resume_offset > 0) {
          curl::handle_setopt(h, resume_from = resume_offset)
          curl::curl_download(u, part_file, handle = h, quiet = TRUE, mode = "ab")
        } else {
          curl::curl_download(u, part_file, handle = h, quiet = TRUE, mode = "wb")
        }
        TRUE
      },
      args = list(u = url, part_file = part_zip, resume_offset = offset),
      supervise = TRUE
    )
    
    last_size <- progress_file_size(part_zip)
    last_time <- Sys.time()
    
    repeat {
      Sys.sleep(0.5)
      size <- progress_file_size(part_zip)
      now <- Sys.time()
      
      dt <- as.numeric(difftime(now, last_time, units = "secs"))
      speed <- if (dt > 0) ((size - last_size) / dt) / MB else 0
      
      cat(sprintf("\rDownloading: %8.1f MB | %6.2f MB/s", size / MB, max(speed, 0)))
      flush.console()
      
      last_size <- size
      last_time <- now
      
      if (!p$is_alive()) break
    }
    cat("\n")
    
    if (!isTRUE(p$get_exit_status() == 0)) {
      err <- tryCatch(p$get_error(), error = function(e) NULL)
      log_warn("Download ended with error.")
      if (!is.null(err)) log_warn(paste0("Underlying error: ", conditionMessage(err)))
      next
    }
    
    if (file.exists(part_curltmp) && !file.exists(part_zip)) {
      log_warn("Finalizing .curltmp -> .part after successful download.")
      ok <- file.rename(part_curltmp, part_zip)
      if (!isTRUE(ok)) {
        ok2 <- file.copy(part_curltmp, part_zip, overwrite = TRUE)
        if (isTRUE(ok2)) file.remove(part_curltmp) else stop_with_context("Could not finalize .curltmp to .part after successful download.")
      }
    }
    
    part_bytes <- if (file.exists(part_zip)) file.info(part_zip)$size else 0
    if (!is.finite(part_bytes) || part_bytes <= 0) {
      log_warn("Download finished but .part is missing/empty. Retrying...")
      next
    }
    
    if (is.finite(expected_bytes) && part_bytes < expected_bytes * min_ok_ratio) {
      log_warn(sprintf(
        "Download appears incomplete (local %.1f MB vs expected %.1f MB). Retrying/resuming...",
        part_bytes / MB, expected_bytes / MB
      ))
      next
    }
    
    if (file.exists(final_zip)) file.remove(final_zip)
    ok_rename <- file.rename(part_zip, final_zip)
    if (!isTRUE(ok_rename)) {
      ok_copy <- file.copy(part_zip, final_zip, overwrite = TRUE)
      if (isTRUE(ok_copy)) file.remove(part_zip) else stop_with_context("Could not finalize download (.part -> final zip).")
    }
    
    final_bytes <- file.info(final_zip)$size
    log_info(sprintf("Download completed: %s", final_zip))
    log_info(sprintf("File size: %.1f MB", final_bytes / MB))
    return(invisible(TRUE))
  }
  
  stop_with_context("Download did not complete after all attempts.")
}

############################
# 5) Unzip (outer + nested)
############################
locate_protocol_dir <- function(root_dir) {
  candidates <- list.dirs(root_dir, recursive = TRUE, full.names = TRUE)
  candidates <- candidates[basename(candidates) == "Protocol"]
  if (length(candidates) == 0) return(NA_character_)
  
  has_dat <- vapply(
    candidates,
    function(p) length(list.files(p, pattern = "^subject\\d+\\.dat$", full.names = TRUE)) > 0,
    logical(1)
  )
  hits <- candidates[has_dat]
  if (length(hits) == 0) return(NA_character_)
  
  root_norm <- normalizePath(root_dir, winslash = "/", mustWork = FALSE)
  hit_norm  <- normalizePath(hits, winslash = "/", mustWork = FALSE)
  depths <- nchar(sub(paste0("^", root_norm, "/?"), "", hit_norm))
  hits[which.min(depths)]
}

unzip_to_dir <- function(zip_path, exdir) {
  dir.create(exdir, recursive = TRUE, showWarnings = FALSE)
  if (!is_writable_dir(exdir)) stop_with_context(sprintf("Extract directory is not writable: %s", exdir))
  
  log_info(paste0("Unzipping: ", zip_path))
  tryCatch(
    utils::unzip(zipfile = zip_path, exdir = exdir),
    error = function(e) stop_with_context("Unzip failed. ZIP may be corrupted/incomplete.", e)
  )
}

ensure_extracted_with_nested_zip <- function(outer_zip, extract_dir) {
  proto <- locate_protocol_dir(extract_dir)
  if (!is.na(proto)) {
    log_info(paste0("Extraction already complete. Protocol found at: ", proto))
    return(invisible(TRUE))
  }
  
  unzip_to_dir(outer_zip, extract_dir)
  
  proto <- locate_protocol_dir(extract_dir)
  if (!is.na(proto)) {
    log_info(paste0("Protocol found after outer unzip at: ", proto))
    return(invisible(TRUE))
  }
  
  nested_zips <- list.files(extract_dir, pattern = "\\.zip$", full.names = TRUE, recursive = TRUE)
  nested_candidate <- nested_zips[grepl("PAMAP2_Dataset\\.zip$", nested_zips)]
  if (length(nested_candidate) == 0 && length(nested_zips) > 0) nested_candidate <- nested_zips
  
  if (length(nested_candidate) == 0) {
    top <- list.files(extract_dir, recursive = FALSE)
    stop_with_context(
      paste0(
        "Outer unzip completed but no nested ZIP found, and Protocol folder is still missing.\n",
        "Top-level extracted entries:\n - ",
        paste(top, collapse = "\n - "),
        "\nInspect: ", extract_dir
      )
    )
  }
  
  log_info("Nested ZIP detected. Unzipping nested dataset ZIP:")
  log_info(paste0(" - ", nested_candidate[1]))
  unzip_to_dir(nested_candidate[1], extract_dir)
  
  proto <- locate_protocol_dir(extract_dir)
  if (is.na(proto)) stop_with_context("Nested unzip completed but Protocol folder still not found. Inspect: " %+% extract_dir)
  
  log_info(paste0("Protocol found after nested unzip at: ", proto))
  invisible(TRUE)
}

############################
# 6) Run download + extract + resolve dirs
############################
download_zip_resumable(pamap2_zip_url, pamap2_zip_path, pamap2_zip_part, attempts = 3, min_ok_ratio = 0.99)
ensure_extracted_with_nested_zip(pamap2_zip_path, dataset_extract_dir)

protocol_dir <- locate_protocol_dir(dataset_extract_dir)
if (is.na(protocol_dir)) stop_with_context("Protocol folder not located after extraction. Inspect: " %+% dataset_extract_dir)

dataset_dir_detected <- dirname(protocol_dir)
optional_dir <- file.path(dataset_dir_detected, "Optional")

log_info("Resolved dataset structure:")
log_info(paste0(" - protocol_dir: ", protocol_dir))
log_info(paste0(" - optional_dir: ", optional_dir, ifelse(dir.exists(optional_dir), "", " (not found)")))

############################
# 7) PAMAP2 schema
############################
activity_map <- tibble::tibble(
  activity_id = c(0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24),
  activity = c(
    "other_transient",
    "lying", "sitting", "standing", "walking", "running", "cycling", "nordic_walking",
    "watching_tv", "computer_work", "car_driving",
    "ascending_stairs", "descending_stairs",
    "vacuum_cleaning", "ironing", "folding_laundry", "house_cleaning", "playing_soccer", "rope_jumping"
  )
)

make_pamap2_colnames <- function() {
  base <- c("timestamp", "activity_id", "heart_rate")
  imu_cols <- c(
    "temp",
    "acc16_x", "acc16_y", "acc16_z",
    "acc6_x",  "acc6_y",  "acc6_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "mag_x",   "mag_y",   "mag_z",
    "orient_w","orient_x","orient_y","orient_z"
  )
  c(base, paste0("hand_", imu_cols), paste0("chest_", imu_cols), paste0("ankle_", imu_cols))
}
pamap2_colnames <- make_pamap2_colnames()
stopifnot(length(pamap2_colnames) == 54)
log_info("Schema ready: 54 columns confirmed.")

############################
# 8) Ingestion
############################
discover_dat_files <- function(dir_path, session_type) {
  if (!dir.exists(dir_path)) return(tibble::tibble())
  files <- list.files(dir_path, pattern = "\\.dat$", full.names = TRUE)
  tibble::tibble(filepath = files) %>%
    mutate(
      session_type = session_type,
      filename = basename(filepath),
      subject_id = stringr::str_extract(tolower(filename), "subject\\d+") %>%
        stringr::str_remove("subject") %>%
        as.integer()
    )
}

file_index <- dplyr::bind_rows(
  discover_dat_files(protocol_dir, "protocol"),
  discover_dat_files(optional_dir, "optional")
) %>%
  arrange(session_type, subject_id, filename)

if (nrow(file_index) == 0) stop_with_context("No input .dat files found. Check protocol_dir/optional_dir above.")

log_info(sprintf("Discovered %d .dat files total.", nrow(file_index)))
print_head(file_index, n = 10, title = "File index (first 10):")

read_pamap2_file <- function(filepath, subject_id, session_type) {
  df <- data.table::fread(
    filepath,
    header = FALSE,
    sep = " ",
    na.strings = c("NaN", "nan"),
    data.table = FALSE,
    showProgress = FALSE
  )
  if (ncol(df) != 54) stop_with_context(sprintf("Invalid column count (%d) in %s (expected 54).", ncol(df), basename(filepath)))
  colnames(df) <- pamap2_colnames
  df %>%
    mutate(
      subject_id = as.integer(subject_id),
      session_type = session_type,
      source_file = basename(filepath),
      row_in_file = dplyr::row_number()
    )
}

log_info("Reading .dat files into memory (this may take a few minutes)...")
data_list <- purrr::pmap(
  list(file_index$filepath, file_index$subject_id, file_index$session_type),
  ~ read_pamap2_file(..1, ..2, ..3)
)

pamap_raw <- dplyr::bind_rows(data_list) %>%
  mutate(
    timestamp   = as.numeric(timestamp),
    activity_id = as.integer(activity_id),
    heart_rate  = as.numeric(heart_rate)
  ) %>%
  left_join(activity_map, by = "activity_id") %>%
  mutate(
    activity = ifelse(is.na(activity), "unknown_activity", activity),
    activity = factor(activity)
  )

log_info(sprintf("Raw dataset loaded: %s rows, %s columns", scales::comma(nrow(pamap_raw)), ncol(pamap_raw)))
print_head(pamap_raw, n = 6, title = "pamap_raw preview:")

############################
# 9) Ingestion audits + Save
############################
calc_median_dt <- function(ts) {
  ts <- ts[is.finite(ts)]
  if (length(ts) < 3) return(NA_real_)
  median(diff(ts), na.rm = TRUE)
}
calc_missing_rate <- function(x) mean(is.na(x))

orientation_cols <- c(
  "hand_orient_w","hand_orient_x","hand_orient_y","hand_orient_z",
  "chest_orient_w","chest_orient_x","chest_orient_y","chest_orient_z",
  "ankle_orient_w","ankle_orient_x","ankle_orient_y","ankle_orient_z"
)

log_info("Building ingestion audit tables...")
audit_by_file <- pamap_raw %>%
  group_by(session_type, subject_id, source_file) %>%
  summarise(
    n_rows = n(),
    ts_min = min(timestamp, na.rm = TRUE),
    ts_max = max(timestamp, na.rm = TRUE),
    median_dt = calc_median_dt(timestamp),
    hr_missing_rate = calc_missing_rate(heart_rate),
    has_unknown_activity = any(activity == "unknown_activity"),
    orientation_var_mean = mean(sapply(across(all_of(orientation_cols), ~ var(.x, na.rm = TRUE)), identity), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(session_type, subject_id, source_file)

write.csv(audit_by_file, file.path(out_ingestion_dir, "ingestion_audit_by_file.csv"), row.names = FALSE)
log_table_preview(audit_by_file, n = 10, title = "Ingestion audit by file (first 10):")

audit_global <- pamap_raw %>%
  summarise(
    rows_total = n(),
    subjects = n_distinct(subject_id),
    files = n_distinct(source_file),
    sessions = n_distinct(session_type),
    hr_missing_rate = mean(is.na(heart_rate)),
    unknown_activity_rows = sum(activity == "unknown_activity")
  )

write.csv(audit_global, file.path(out_ingestion_dir, "ingestion_audit_global.csv"), row.names = FALSE)
log_table_preview(audit_global, n = 1, title = "Ingestion audit (global):")

saveRDS(pamap_raw, file.path(out_processed_dir, "pamap_raw_all.rds"))
saveRDS(pamap_raw %>% dplyr::filter(session_type == "protocol"), file.path(out_processed_dir, "pamap_raw_protocol.rds"))
saveRDS(pamap_raw %>% dplyr::filter(session_type == "optional"), file.path(out_processed_dir, "pamap_raw_optional.rds"))

log_info("STEP 01 outputs saved:")
log_info(paste0(" - ", file.path(out_processed_dir, "pamap_raw_all.rds")))
log_info(paste0(" - ", file.path(out_ingestion_dir, "ingestion_audit_global.csv")))
log_info(paste0(" - ", file.path(out_ingestion_dir, "ingestion_audit_by_file.csv")))
log_step("STEP 01 completed successfully")

##############################################
# STEP 02 RAW EDA (BEFORE cleaning/imputation)
##############################################
log_step("STEP 02 RAW EDA (before cleaning/imputation)")

############################
# 10) EDA paths
############################
in_processed <- file.path(project_root, "data", "processed", "pamap_raw_all.rds")
if (!file.exists(in_processed)) stop_with_context("pamap_raw_all.rds not found. Run STEP 01 first.")

eda_out_dir <- file.path(project_root, "outputs", "eda_raw")
dir.create(eda_out_dir, showWarnings = FALSE, recursive = TRUE)

eda_fig_dir <- file.path(eda_out_dir, "figures")
dir.create(eda_fig_dir, showWarnings = FALSE, recursive = TRUE)

write_csv_safe <- function(df, path) {
  readr::write_csv(df, path)
  log_info(paste0("Saved: ", path))
}

save_plot <- function(p, filename, width = 12, height = 7, dpi = 150) {
  path <- file.path(eda_fig_dir, filename)
  ggplot2::ggsave(path, plot = p, width = width, height = height, dpi = dpi)
  log_info(paste0("Saved figure: ", path))
}

median_dt2 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  median(diff(x), na.rm = TRUE)
}
iqr_dt2 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 5) return(NA_real_)
  stats::IQR(diff(x), na.rm = TRUE)
}
missing_rate2 <- function(x) mean(is.na(x))

############################
# 11) Load raw data
############################
pamap <- readRDS(in_processed) %>%
  mutate(
    timestamp = as.numeric(timestamp),
    activity_id = as.integer(activity_id),
    subject_id = as.integer(subject_id),
    session_type = factor(session_type),
    activity = as.character(activity)
  )

log_info(sprintf("Loaded pamap_raw_all.rds: %s rows, %s cols", scales::comma(nrow(pamap)), ncol(pamap)))

############################
# 12) Global snapshot
############################
summary_global <- pamap %>%
  summarise(
    rows_total = n(),
    subjects = n_distinct(subject_id),
    files = n_distinct(source_file),
    sessions = n_distinct(session_type),
    activities = n_distinct(activity),
    hr_missing_rate = mean(is.na(heart_rate)),
    activity0_rate = mean(activity_id == 0)
  )

write_csv_safe(summary_global, file.path(eda_out_dir, "global_summary.csv"))
log_table_preview(summary_global, n = 1, title = "EDA Global snapshot:")

############################
# 13) Coverage by subject/session
############################
by_subject <- pamap %>%
  count(subject_id, session_type, name = "rows") %>%
  arrange(subject_id, session_type)

write_csv_safe(by_subject, file.path(eda_out_dir, "rows_by_subject_session.csv"))
log_table_preview(by_subject, n = 20, title = "Rows by subject/session (first 20):")

p_rows <- by_subject %>%
  ggplot(aes(x = factor(subject_id), y = rows, fill = session_type)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "PAMAP2 Raw Data Coverage by Subject and Session Type",
    x = "Subject ID",
    y = "Number of Rows",
    fill = "Session Type"
  ) +
  theme_minimal(base_size = 12)

print(p_rows)
save_plot(p_rows, "rows_by_subject_session.png")

############################
# 14) Activity distribution (top preview)
############################
activity_overall <- pamap %>%
  count(activity_id, activity, name = "rows") %>%
  mutate(pct = rows / sum(rows)) %>%
  arrange(desc(rows))

write_csv_safe(activity_overall, file.path(eda_out_dir, "activity_distribution_overall.csv"))
log_table_preview(activity_overall, n = 15, title = "Activity distribution (top 15):")

p_act <- activity_overall %>%
  mutate(activity = factor(activity, levels = activity[order(rows)])) %>%
  ggplot(aes(x = activity, y = rows)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Activity Distribution (Raw Rows)",
    x = "Activity",
    y = "Number of Rows"
  ) +
  theme_minimal(base_size = 12)

save_plot(p_act, "activity_distribution_overall.png")
print(p_act)

############################
# 15) Missingness profiling (top preview)
############################
orientation_cols2 <- c(
  "hand_orient_w","hand_orient_x","hand_orient_y","hand_orient_z",
  "chest_orient_w","chest_orient_x","chest_orient_y","chest_orient_z",
  "ankle_orient_w","ankle_orient_x","ankle_orient_y","ankle_orient_z"
)

acc6_cols <- c(
  "hand_acc6_x","hand_acc6_y","hand_acc6_z",
  "chest_acc6_x","chest_acc6_y","chest_acc6_z",
  "ankle_acc6_x","ankle_acc6_y","ankle_acc6_z"
)

acc16_cols <- c(
  "hand_acc16_x","hand_acc16_y","hand_acc16_z",
  "chest_acc16_x","chest_acc16_y","chest_acc16_z",
  "ankle_acc16_x","ankle_acc16_y","ankle_acc16_z"
)

gyro_cols <- c(
  "hand_gyro_x","hand_gyro_y","hand_gyro_z",
  "chest_gyro_x","chest_gyro_y","chest_gyro_z",
  "ankle_gyro_x","ankle_gyro_y","ankle_gyro_z"
)

mag_cols <- c(
  "hand_mag_x","hand_mag_y","hand_mag_z",
  "chest_mag_x","chest_mag_y","chest_mag_z",
  "ankle_mag_x","ankle_mag_y","ankle_mag_z"
)

temp_cols <- c("hand_temp", "chest_temp", "ankle_temp")

all_cols <- c("heart_rate", temp_cols, acc16_cols, acc6_cols, gyro_cols, mag_cols, orientation_cols2)
missing_cols <- setdiff(all_cols, names(pamap))
if (length(missing_cols) > 0) stop_with_context("Missing expected columns: " %+% paste(missing_cols, collapse = ", "))

missing_global <- tibble::tibble(
  feature = all_cols,
  missing_rate = purrr::map_dbl(all_cols, ~ mean(is.na(pamap[[.x]])))
) %>%
  arrange(desc(missing_rate))

write_csv_safe(missing_global, file.path(eda_out_dir, "missingness_global_by_feature.csv"))
log_table_preview(missing_global, n = 15, title = "Missingness by feature (top 15):")

p_miss <- missing_global %>%
  mutate(feature = factor(feature, levels = feature[order(missing_rate)])) %>%
  ggplot(aes(x = feature, y = missing_rate)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Missingness by Feature (Raw Data)",
    x = "Feature",
    y = "Missing Rate"
  ) +
  theme_minimal(base_size = 12)

save_plot(p_miss, "missingness_by_feature.png", width = 12, height = 10)
print(p_miss)

###############################################################################
# 16) CHARACTERIZATION OF SAMPLING RATE (AUDIT & VISUALIZATION)
###############################################################################
log_info("Starting Step 16: Sampling rate audit...")

# 1. Calculation: Group by file to check temporal consistency
sampling_by_file <- pamap %>%
  group_by(session_type, subject_id, source_file) %>%
  summarise(
    median_dt = median_dt2(timestamp),
    iqr_dt    = iqr_dt2(timestamp),
    approx_hz = ifelse(is.finite(median_dt) & median_dt > 0, 1 / median_dt, NA_real_),
    .groups   = "drop"
  )

# 2. Statistical Audit: Console summary for the report narrative
summary_stats <- sampling_by_file %>%
  summarise(
    Min_Hz       = min(approx_hz, na.rm = TRUE),
    Max_Hz       = max(approx_hz, na.rm = TRUE),
    Files_at_100 = sum(approx_hz >= 99 & approx_hz <= 101, na.rm = TRUE),
    Anomalies    = sum(approx_hz < 90, na.rm = TRUE)
  )

log_info("Sampling frequency quick stats:")
print(summary_stats)

# Guard-rail check: identify any files below 90Hz
sampling_anomalies <- sampling_by_file %>% filter(approx_hz < 90)
if(nrow(sampling_anomalies) > 0) {
  log_warn("Anomalous sampling rates detected in:")
  print(sampling_anomalies)
}

# 3. Final Visualization: Clean categorical bar chart for the PDF report
p_hz_final <- sampling_by_file %>%
  mutate(hz_label = factor(round(approx_hz, 1))) %>% # Clean floating point noise
  ggplot(aes(x = hz_label)) +
  geom_bar(fill = "steelblue", color = "white", width = 0.4) +
  theme_minimal(base_size = 12) +
  labs(
    title    = "Validated Sampling Frequency (PAMAP2)",
    subtitle = "Consistency check: 100 Hz nominal rate across all subjects",
    x        = "Approximate Frequency (Hz)",
    y        = "Number of Files"
  )

# 4. Export Artifacts: Save table and plot
write_csv_safe(sampling_by_file, file.path(eda_out_dir, "sampling_rate_by_file.csv"))
save_plot(p_hz_final, "sampling_frequency_distribution.png")

# Display final plot in console/Rmd
print(p_hz_final)

############################
# 17) Acc saturation proxy
############################
g_ms2 <- 9.81
limit_acc6  <- 6  * g_ms2
limit_acc16 <- 16 * g_ms2

# "Near saturation" margin (tunable): 5% of 1g
eps <- 0.05 * g_ms2

sat_summary <- pamap %>%
  summarise(
    acc6_sat_rate = mean(
      rowSums(dplyr::across(all_of(acc6_cols),  ~ abs(.x) >= (limit_acc6  - eps)), na.rm = TRUE) > 0,
      na.rm = TRUE
    ),
    acc16_sat_rate = mean(
      rowSums(dplyr::across(all_of(acc16_cols), ~ abs(.x) >= (limit_acc16 - eps)), na.rm = TRUE) > 0,
      na.rm = TRUE
    )
  )

write_csv_safe(sat_summary, file.path(eda_out_dir, "acc_saturation_global.csv"))
log_table_preview(sat_summary, n = 1, title = "Acceleration saturation proxy (global):")

# 18) Orientation variance proxy (robust dplyr version)
############################
orientation_var <- pamap %>%
  group_by(subject_id, session_type) %>%
  summarise(
    across(all_of(orientation_cols2), ~ stats::var(.x, na.rm = TRUE), .names = "var_{.col}"),
    .groups = "drop"
  ) %>%
  mutate(
    mean_orientation_variance = rowMeans(dplyr::select(., dplyr::starts_with("var_")), na.rm = TRUE)
  ) %>%
  select(subject_id, session_type, mean_orientation_variance) %>%
  arrange(subject_id, session_type)

write_csv_safe(orientation_var, file.path(eda_out_dir, "orientation_variance_by_subject_session.csv"))
log_table_preview(orientation_var, n = 20, title = "Orientation variance proxy (first 20 rows):")

p_orient <- orientation_var %>%
  ggplot(aes(x = factor(subject_id), y = mean_orientation_variance, fill = session_type)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Orientation Channels: Mean Variance by Subject and Session Type",
    x = "Subject ID",
    y = "Mean Variance (Orientation Channels)",
    fill = "Session Type"
  ) +
  theme_minimal(base_size = 12)

save_plot(p_orient, "orientation_variance_by_subject_session.png")

print(p_orient)

# Orientation channels were constant for subject 101
pamap %>%
  filter(subject_id == 101) %>%
  summarise(across(all_of(orientation_cols2), ~ dplyr::n_distinct(.x, na.rm = TRUE), .names = "unique_{.col}")) %>%
  tidyr::pivot_longer(everything(), names_to = "channel", values_to = "n_unique") %>%
  arrange(n_unique)



############################
# 19) Activity 0 prevalence
############################
activity0_by_subject <- pamap %>%
  group_by(subject_id, session_type) %>%
  summarise(
    rows = n(),
    activity0_rate = mean(activity_id == 0),
    .groups = "drop"
  )

write_csv_safe(activity0_by_subject, file.path(eda_out_dir, "activity0_rate_by_subject_session.csv"))
log_table_preview(activity0_by_subject, n = 20, title = "Activity ID=0 prevalence by subject/session (first 20):")

p_act0 <- activity0_by_subject %>%
  ggplot(aes(x = factor(subject_id), y = activity0_rate, fill = session_type)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Share of Activity ID = 0 (Other/Transient) by Subject and Session Type",
    x = "Subject ID",
    y = "Rate of Activity 0",
    fill = "Session Type"
  ) +
  theme_minimal(base_size = 12)

save_plot(p_act0, "activity0_rate_by_subject_session.png")
print(p_act0)

############################
# 20) Executive markdown snapshot + final console summary
############################
md_path <- file.path(eda_out_dir, "eda_raw_summary.md")
md_lines <- c(
  "# PAMAP2 Raw EDA Summary",
  "",
  "## Global Snapshot",
  sprintf("- Total rows: %s", scales::comma(summary_global$rows_total)),
  sprintf("- Subjects: %s", summary_global$subjects),
  sprintf("- Session types: %s", summary_global$sessions),
  sprintf("- Unique activity labels: %s", summary_global$activities),
  sprintf("- Heart rate missingness (global): %s", scales::percent(summary_global$hr_missing_rate, accuracy = 0.1)),
  sprintf("- Activity ID = 0 share (global): %s", scales::percent(summary_global$activity0_rate, accuracy = 0.1)),
  "",
  "## Key EDA Artifacts Produced",
  "- Coverage: rows_by_subject_session.csv + rows_by_subject_session.png",
  "- Activity distribution: activity_distribution_overall.csv + activity_distribution_overall.png",
  "- Missingness: missingness_global_by_feature.csv + missingness_by_feature.png",
  "- Sampling frequency: sampling_rate_by_file.csv + sampling_frequency_distribution.png",
  "- Acc saturation proxy: acc_saturation_global.csv",
  "- Orientation variance proxy: orientation_variance_by_subject_session.csv + orientation_variance_by_subject_session.png",
  "- Activity 0 prevalence: activity0_rate_by_subject_session.csv + activity0_rate_by_subject_session.png"
)
writeLines(md_lines, md_path)
log_info(paste0("Saved: ", md_path))
log_step("STEP 02 completed successfully")
log_info("EDA outputs directory:")
log_info(eda_out_dir)

##############################################
# STEP 03  CLEANING & FEATURE SELECTION (Protocol only)
##############################################
log_step("STEP 03 CLEANING & FEATURE SELECTION (Protocol only, no leakage)")

############################
# 21) Load raw protocol dataset (canonical input for cleaning)
############################
clean_in_path <- file.path(project_root, "data", "processed", "pamap_raw_protocol.rds")
if (!file.exists(clean_in_path)) stop_with_context("pamap_raw_protocol.rds not found. Run STEP 01 first.")

pamap_raw_for_clean <- readRDS(clean_in_path)
log_info(sprintf(
  "Loaded raw protocol for cleaning: %s rows, %s cols",
  scales::comma(nrow(pamap_raw_for_clean)), ncol(pamap_raw_for_clean)
))

############################
# 22) Cleaning configuration (explicit, report-friendly)
############################
# IMPORTANT: Keep these flags explicit because they will be referenced in the final report.
USE_HEART_RATE_FOR_MODEL <- FALSE   # default: do NOT use HR due to asynchronous sampling and heavy missingness
FILL_HEART_RATE_LOCF     <- TRUE    # used only for audit; if USE_HEART_RATE_FOR_MODEL=TRUE, we fill forward only (no look-ahead)

DROP_ACTIVITY_0          <- TRUE    # activity_id == 0 = "other/transient"
DROP_UNKNOWN_ACTIVITY    <- TRUE    # safety guard
DROP_ORIENTATION         <- TRUE    # documented invalid
DROP_ACC6                <- TRUE    # 6g may saturate; prefer 16g
DROP_TEMP                <- FALSE   # keep temps for now (can be removed later if unhelpful)

log_info("Cleaning configuration:")
log_info(paste0(" - USE_HEART_RATE_FOR_MODEL: ", USE_HEART_RATE_FOR_MODEL))
log_info(paste0(" - FILL_HEART_RATE_LOCF:     ", FILL_HEART_RATE_LOCF))
log_info(paste0(" - DROP_ACTIVITY_0:          ", DROP_ACTIVITY_0))
log_info(paste0(" - DROP_UNKNOWN_ACTIVITY:    ", DROP_UNKNOWN_ACTIVITY))
log_info(paste0(" - DROP_ORIENTATION:         ", DROP_ORIENTATION))
log_info(paste0(" - DROP_ACC6:                ", DROP_ACC6))
log_info(paste0(" - DROP_TEMP:                ", DROP_TEMP))

############################
# 23) Column groups (re-declare here to keep Step 03 self-contained)
############################
orientation_cols3 <- c(
  "hand_orient_w","hand_orient_x","hand_orient_y","hand_orient_z",
  "chest_orient_w","chest_orient_x","chest_orient_y","chest_orient_z",
  "ankle_orient_w","ankle_orient_x","ankle_orient_y","ankle_orient_z"
)

acc6_cols3 <- c(
  "hand_acc6_x","hand_acc6_y","hand_acc6_z",
  "chest_acc6_x","chest_acc6_y","chest_acc6_z",
  "ankle_acc6_x","ankle_acc6_y","ankle_acc6_z"
)

temp_cols3 <- c("hand_temp", "chest_temp", "ankle_temp")

# sanity checks (do not hard-stop on optional columns; warn instead)
missing_required <- setdiff(c("timestamp","activity_id","subject_id","source_file","session_type","activity"), names(pamap_raw_for_clean))
if (length(missing_required) > 0) stop_with_context("Missing required columns: " %+% paste(missing_required, collapse = ", "))

############################
# 24) Cleaning output paths
############################
clean_out_dir <- file.path(project_root, "outputs", "cleaning")
dir.create(clean_out_dir, showWarnings = FALSE, recursive = TRUE)

clean_processed_path <- file.path(project_root, "data", "processed", "pamap_clean_protocol.rds")
clean_meta_path      <- file.path(project_root, "outputs", "cleaning", "cleaning_summary_protocol.csv")

############################
# 25) Core cleaning (use data.table for speed and memory efficiency)
############################
DT <- data.table::as.data.table(pamap_raw_for_clean)

rows_before <- nrow(DT)
cols_before <- ncol(DT)

# Basic validity filters
DT <- DT[is.finite(timestamp) & !is.na(activity_id) & !is.na(subject_id)]
rows_after_basic <- nrow(DT)

removed_basic <- rows_before - rows_after_basic
if (removed_basic > 0) log_info(sprintf("Removed %s rows due to invalid timestamp/activity_id/subject_id.", scales::comma(removed_basic)))

# Drop activity 0 (other/transient)
removed_activity0 <- 0
if (DROP_ACTIVITY_0) {
  removed_activity0 <- DT[activity_id == 0, .N]
  DT <- DT[activity_id != 0]
  log_info(sprintf("Removed activity_id==0 rows: %s", scales::comma(removed_activity0)))
}

# Drop unknown activity rows (guard)
removed_unknown <- 0
if (DROP_UNKNOWN_ACTIVITY && "activity" %in% names(DT)) {
  removed_unknown <- DT[as.character(activity) == "unknown_activity", .N]
  DT <- DT[as.character(activity) != "unknown_activity"]
  if (removed_unknown > 0) log_warn(sprintf("Removed unknown_activity rows: %s", scales::comma(removed_unknown)))
}

# Ensure deterministic ordering per file (needed later for windowing)
data.table::setorder(DT, subject_id, source_file, timestamp, row_in_file)

# Heart rate audit + optional forward fill (LOCF only, no look-ahead)
hr_missing_before <- if ("heart_rate" %in% names(DT)) mean(is.na(DT$heart_rate)) else NA_real_
if ("heart_rate" %in% names(DT)) {
  DT[, hr_missing_raw := is.na(heart_rate)]
}

if ("heart_rate" %in% names(DT) && isTRUE(FILL_HEART_RATE_LOCF)) {
  # forward fill only within each file
  DT[, heart_rate_locf := data.table::nafill(heart_rate, type = "locf"), by = .(subject_id, source_file)]
  hr_missing_after <- mean(is.na(DT$heart_rate_locf))
  log_info(sprintf("Heart rate missingness: before=%.2f%% | after LOCF=%.2f%%",
                   100 * hr_missing_before, 100 * hr_missing_after))
} else {
  hr_missing_after <- NA_real_
}

# Drop orientation channels (documented invalid)
dropped_cols <- character(0)
if (DROP_ORIENTATION) {
  cols_to_drop <- intersect(orientation_cols3, names(DT))
  if (length(cols_to_drop) > 0) {
    DT[, (cols_to_drop) := NULL]
    dropped_cols <- c(dropped_cols, cols_to_drop)
    log_info(sprintf("Dropped orientation columns: %d", length(cols_to_drop)))
  } else {
    log_warn("No orientation columns found to drop (already removed or schema mismatch).")
  }
}

# Drop acc6 (6g) channels (prefer acc16)
if (DROP_ACC6) {
  cols_to_drop <- intersect(acc6_cols3, names(DT))
  if (length(cols_to_drop) > 0) {
    DT[, (cols_to_drop) := NULL]
    dropped_cols <- c(dropped_cols, cols_to_drop)
    log_info(sprintf("Dropped acc6 columns: %d", length(cols_to_drop)))
  } else {
    log_warn("No acc6 columns found to drop (already removed or schema mismatch).")
  }
}

# Optionally drop temperature channels
if (DROP_TEMP) {
  cols_to_drop <- intersect(temp_cols3, names(DT))
  if (length(cols_to_drop) > 0) {
    DT[, (cols_to_drop) := NULL]
    dropped_cols <- c(dropped_cols, cols_to_drop)
    log_info(sprintf("Dropped temperature columns: %d", length(cols_to_drop)))
  }
}

# If HR is NOT used for model, remove HR columns to avoid accidental leakage/usage later
if (!USE_HEART_RATE_FOR_MODEL) {
  hr_cols_drop <- intersect(c("heart_rate", "heart_rate_locf", "hr_missing_raw"), names(DT))
  if (length(hr_cols_drop) > 0) {
    DT[, (hr_cols_drop) := NULL]
    dropped_cols <- c(dropped_cols, hr_cols_drop)
    log_info("Heart rate excluded from modeling dataset (columns removed).")
  }
} else {
  # If HR is used, keep heart_rate_locf as the modeling HR channel, and keep missing flag
  if ("heart_rate_locf" %in% names(DT)) {
    DT[, heart_rate := NULL]  # avoid confusion: keep only filled version
    data.table::setnames(DT, "heart_rate_locf", "heart_rate")
    log_info("Heart rate included: using LOCF-filled heart_rate + hr_missing_raw flag.")
  } else {
    log_warn("USE_HEART_RATE_FOR_MODEL=TRUE but heart_rate_locf not found; HR will remain as-is.")
  }
}

# Final type hygiene
DT[, activity_id := as.integer(activity_id)]
DT[, subject_id  := as.integer(subject_id)]
DT[, session_type := as.factor(session_type)]
DT[, activity := as.factor(as.character(activity))]

rows_after <- nrow(DT)
cols_after <- ncol(DT)

log_info(sprintf("Cleaning result: %s -> %s rows | %s -> %s cols",
                 scales::comma(rows_before), scales::comma(rows_after), cols_before, cols_after))

############################
# 26) Save cleaned dataset + cleaning summary artifacts
############################
pamap_clean <- as.data.frame(DT)
saveRDS(pamap_clean, clean_processed_path)

log_info(paste0("Saved cleaned dataset: ", clean_processed_path))

cleaning_summary <- tibble::tibble(
  metric = c(
    "rows_before",
    "rows_after_basic_validity",
    "rows_removed_basic_validity",
    "rows_removed_activity0",
    "rows_removed_unknown_activity",
    "rows_after_final",
    "cols_before",
    "cols_after_final",
    "hr_missing_before_raw",
    "hr_missing_after_locf_audit",
    "use_heart_rate_for_model",
    "fill_heart_rate_locf",
    "drop_activity0",
    "drop_unknown_activity",
    "drop_orientation",
    "drop_acc6",
    "drop_temp",
    "dropped_columns_count"
  ),
  value = c(
    rows_before,
    rows_after_basic,
    removed_basic,
    removed_activity0,
    removed_unknown,
    rows_after,
    cols_before,
    cols_after,
    hr_missing_before,
    hr_missing_after,
    USE_HEART_RATE_FOR_MODEL,
    FILL_HEART_RATE_LOCF,
    DROP_ACTIVITY_0,
    DROP_UNKNOWN_ACTIVITY,
    DROP_ORIENTATION,
    DROP_ACC6,
    DROP_TEMP,
    length(unique(dropped_cols))
  )
)

readr::write_csv(cleaning_summary, clean_meta_path)
log_info(paste0("Saved cleaning summary: ", clean_meta_path))

# Optional: write dropped columns list for traceability
dropped_cols_path <- file.path(clean_out_dir, "dropped_columns_protocol.txt")
writeLines(sort(unique(dropped_cols)), dropped_cols_path)
log_info(paste0("Saved dropped columns list: ", dropped_cols_path))

# Console preview: activity distribution after cleaning
act_after <- pamap_clean %>%
  dplyr::count(activity_id, activity, name = "rows") %>%
  dplyr::mutate(pct = rows / sum(rows)) %>%
  dplyr::arrange(dplyr::desc(rows))

log_table_preview(act_after, n = 15, title = "Activity distribution AFTER cleaning (top 15):")

log_step("STEP 03 completed successfully")

##############################################
# STEP 04 POST-CLEAN EDA VALIDATION + SUBJECT-INDEPENDENT SPLIT (Protocol only)
##############################################
log_step("STEP 04 POST-CLEAN EDA VALIDATION + SUBJECT-INDEPENDENT SPLIT (Protocol only)")

############################
# 27) Load cleaned dataset
############################
clean_path <- file.path(project_root, "data", "processed", "pamap_clean_protocol.rds")
if (!file.exists(clean_path)) stop_with_context("pamap_clean_protocol.rds not found. Run STEP 03 first.")

pamap_clean <- readRDS(clean_path)

log_info(sprintf(
  "Loaded pamap_clean_protocol.rds: %s rows, %s cols",
  scales::comma(nrow(pamap_clean)), ncol(pamap_clean)
))
print_head(pamap_clean, n = 6, title = "pamap_clean preview:")

############################
# 28) Post-clean outputs dirs
############################
post_out_dir <- file.path(project_root, "outputs", "post_clean_eda")
dir.create(post_out_dir, showWarnings = FALSE, recursive = TRUE)

post_fig_dir <- file.path(post_out_dir, "figures")
dir.create(post_fig_dir, showWarnings = FALSE, recursive = TRUE)

write_csv_safe2 <- function(df, path) {
  readr::write_csv(df, path)
  log_info(paste0("Saved: ", path))
}

save_plot2 <- function(p, filename, width = 12, height = 7, dpi = 150) {
  path <- file.path(post_fig_dir, filename)
  ggplot2::ggsave(path, plot = p, width = width, height = height, dpi = dpi)
  log_info(paste0("Saved figure: ", path))
}

############################
# 29) Guardrails: Protocol-only + no forbidden columns/labels
############################
if (!("session_type" %in% names(pamap_clean))) stop_with_context("session_type missing from cleaned dataset.")
session_levels <- unique(as.character(pamap_clean$session_type))
if (any(session_levels != "protocol")) {
  stop_with_context(
    paste0(
      "Protocol-only guardrail failed. Found session_type values: ",
      paste(session_levels, collapse = ", "),
      "\nEnsure Step 01 ingests Protocol only and Step 03 writes pamap_clean_protocol.rds accordingly."
    )
  )
}
log_info("Guardrail OK: Protocol-only dataset confirmed.")

# Ensure activity_id==0 is gone
if ("activity_id" %in% names(pamap_clean)) {
  n_act0 <- sum(pamap_clean$activity_id == 0, na.rm = TRUE)
  if (n_act0 > 0) stop_with_context(sprintf("Cleaning validation failed: found %s rows with activity_id==0.", scales::comma(n_act0)))
  log_info("Validation OK: activity_id==0 not present.")
} else {
  stop_with_context("activity_id missing from cleaned dataset.")
}

# Ensure orientation columns are gone
orientation_cols_check <- c(
  "hand_orient_w","hand_orient_x","hand_orient_y","hand_orient_z",
  "chest_orient_w","chest_orient_x","chest_orient_y","chest_orient_z",
  "ankle_orient_w","ankle_orient_x","ankle_orient_y","ankle_orient_z"
)
present_orientation <- intersect(orientation_cols_check, names(pamap_clean))
if (length(present_orientation) > 0) {
  stop_with_context(
    paste0(
      "Cleaning validation failed: orientation columns still present: ",
      paste(present_orientation, collapse = ", ")
    )
  )
}
log_info("Validation OK: orientation columns not present.")

# Ensure acc6 columns are gone (if I decided to drop them)
acc6_cols_check <- c(
  "hand_acc6_x","hand_acc6_y","hand_acc6_z",
  "chest_acc6_x","chest_acc6_y","chest_acc6_z",
  "ankle_acc6_x","ankle_acc6_y","ankle_acc6_z"
)
present_acc6 <- intersect(acc6_cols_check, names(pamap_clean))
if (length(present_acc6) > 0) {
  log_warn(paste0(
    "acc6 columns are present in cleaned dataset: ",
    paste(present_acc6, collapse = ", "),
    "\nThis is OK only if you intentionally kept acc6. Otherwise, set DROP_ACC6=TRUE in STEP 03."
  ))
} else {
  log_info("Validation OK: acc6 columns not present.")
}

############################
# 30) Post-clean global snapshot + missingness
############################
required_core <- c("timestamp","activity_id","activity","subject_id","source_file","session_type")
missing_core <- setdiff(required_core, names(pamap_clean))
if (length(missing_core) > 0) stop_with_context(paste0("Missing core columns: ", paste(missing_core, collapse = ", ")))

post_global <- pamap_clean %>%
  summarise(
    rows_total = n(),
    subjects = n_distinct(subject_id),
    files = n_distinct(source_file),
    activities = n_distinct(activity),
    any_na_in_core = any(is.na(timestamp) | is.na(activity_id) | is.na(subject_id) | is.na(activity)),
    pct_rows_with_any_na_feature = mean(!complete.cases(.))
  )

write_csv_safe2(post_global, file.path(post_out_dir, "post_clean_global_summary.csv"))
log_table_preview(post_global, n = 1, title = "Post-clean global snapshot:")

# Missingness by feature (post-clean)
feature_cols <- setdiff(names(pamap_clean), c("activity","activity_id","subject_id","source_file","session_type","row_in_file"))
missing_post <- tibble::tibble(
  feature = feature_cols,
  missing_rate = purrr::map_dbl(feature_cols, ~ mean(is.na(pamap_clean[[.x]])))
) %>%
  arrange(desc(missing_rate))

write_csv_safe2(missing_post, file.path(post_out_dir, "post_clean_missingness_by_feature.csv"))
log_table_preview(missing_post, n = 15, title = "Post-clean missingness (top 15 features):")

p_miss_post <- missing_post %>%
  mutate(feature = factor(feature, levels = feature[order(missing_rate)])) %>%
  ggplot(aes(x = feature, y = missing_rate)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Post-clean Missingness by Feature (Protocol Only)",
    x = "Feature",
    y = "Missing Rate"
  ) +
  theme_minimal(base_size = 12)

save_plot2(p_miss_post, "post_clean_missingness_by_feature.png", width = 12, height = 10)
print(p_miss_post)

############################
# 31) Post-clean activity distribution + subject coverage
############################
act_dist_post <- pamap_clean %>%
  count(activity_id, activity, name = "rows") %>%
  mutate(pct = rows / sum(rows)) %>%
  arrange(desc(rows))

write_csv_safe2(act_dist_post, file.path(post_out_dir, "post_clean_activity_distribution_overall.csv"))
log_table_preview(act_dist_post, n = 15, title = "Post-clean activity distribution (top 15):")

p_act_post <- act_dist_post %>%
  mutate(activity = factor(activity, levels = activity[order(rows)])) %>%
  ggplot(aes(x = activity, y = rows)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Post-clean Activity Distribution (Raw Rows, Protocol Only)",
    x = "Activity",
    y = "Rows"
  ) +
  theme_minimal(base_size = 12)

save_plot2(p_act_post, "post_clean_activity_distribution_overall.png")
print(p_act_post)

rows_by_subject <- pamap_clean %>%
  count(subject_id, name = "rows") %>%
  arrange(desc(rows))

write_csv_safe2(rows_by_subject, file.path(post_out_dir, "post_clean_rows_by_subject.csv"))
log_table_preview(rows_by_subject, n = 20, title = "Post-clean rows by subject:")

p_rows_subj <- rows_by_subject %>%
  mutate(subject_id = factor(subject_id, levels = subject_id[order(rows)])) %>%
  ggplot(aes(x = subject_id, y = rows)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Post-clean Rows by Subject (Protocol Only)",
    x = "Subject ID",
    y = "Rows"
  ) +
  theme_minimal(base_size = 12)

save_plot2(p_rows_subj, "post_clean_rows_by_subject.png")

print(p_rows_subj)

# Activity coverage per subject (how many activities appear at least once)
activity_by_subject <- pamap_clean %>%
  count(subject_id, activity_id, activity, name = "rows") %>%
  group_by(subject_id) %>%
  summarise(
    n_activities_present = sum(rows > 0),
    total_rows = sum(rows),
    .groups = "drop"
  ) %>%
  arrange(desc(n_activities_present), desc(total_rows))

write_csv_safe2(activity_by_subject, file.path(post_out_dir, "post_clean_activity_coverage_by_subject.csv"))
log_table_preview(activity_by_subject, n = 20, title = "Post-clean activity coverage by subject:")

p_cov <- activity_by_subject %>%
  mutate(subject_id = factor(subject_id, levels = subject_id[order(n_activities_present)])) %>%
  ggplot(aes(x = subject_id, y = n_activities_present)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Number of Activities Present by Subject (Protocol Only)",
    x = "Subject ID",
    y = "Activities Present"
  ) +
  theme_minimal(base_size = 12)

save_plot2(p_cov, "post_clean_activity_coverage_by_subject.png")
print(p_cov)

############################
# 32) Subject-independent split design (train/val/test + LOSO folds within train)
############################
log_step("Designing subject-independent split (no leakage)")

# Exclude extremely small subjects from modeling (quality guardrail)
MIN_ROWS_PER_SUBJECT <- 50000

eligible_subjects <- rows_by_subject %>%
  mutate(is_eligible = rows >= MIN_ROWS_PER_SUBJECT)

excluded_subjects <- eligible_subjects %>% filter(!is_eligible)
if (nrow(excluded_subjects) > 0) {
  log_warn("Excluding subjects with insufficient rows (will NOT be used in modeling):")
  print(excluded_subjects)
}

eligible_subject_ids <- eligible_subjects %>%
  filter(is_eligible) %>%
  pull(subject_id) %>%
  sort()

if (length(eligible_subject_ids) < 5) {
  stop_with_context(
    paste0(
      "Not enough eligible subjects after applying MIN_ROWS_PER_SUBJECT=",
      MIN_ROWS_PER_SUBJECT,
      ". Eligible subjects: ",
      paste(eligible_subject_ids, collapse = ", ")
    )
  )
}

# Choose TEST and VALIDATION subjects deterministically:
# - Prefer subjects with maximum activity coverage (more representative)
# - Tie-break by higher row count
elig_cov <- activity_by_subject %>%
  filter(subject_id %in% eligible_subject_ids) %>%
  left_join(rows_by_subject, by = "subject_id") %>%
  arrange(desc(n_activities_present), desc(rows))

test_subject <- elig_cov$subject_id[1]
val_subject  <- elig_cov$subject_id[2]

train_subjects <- setdiff(eligible_subject_ids, c(test_subject, val_subject))

log_info(paste0("Eligible subjects (n=", length(eligible_subject_ids), "): ", paste(eligible_subject_ids, collapse = ", ")))
log_info(paste0("Selected TEST subject: ", test_subject))
log_info(paste0("Selected VAL  subject: ", val_subject))
log_info(paste0("TRAIN subjects (n=", length(train_subjects), "): ", paste(train_subjects, collapse = ", ")))

# Create LOSO folds for internal CV on TRAIN only (each fold leaves one subject out)
loso_folds <- lapply(train_subjects, function(holdout_subj) {
  list(
    fold_id = paste0("LOSO_", holdout_subj),
    train_subjects = setdiff(train_subjects, holdout_subj),
    holdout_subject = holdout_subj
  )
})
names(loso_folds) <- vapply(loso_folds, `[[`, character(1), "fold_id")

# Build split datasets (still row-level; windowing comes later)
train_df <- pamap_clean %>% filter(subject_id %in% train_subjects)
val_df   <- pamap_clean %>% filter(subject_id == val_subject)
test_df  <- pamap_clean %>% filter(subject_id == test_subject)

log_info(sprintf("Split sizes (rows): train=%s | val=%s | test=%s",
                 scales::comma(nrow(train_df)), scales::comma(nrow(val_df)), scales::comma(nrow(test_df))))

# Save split artifacts (data + manifest + fold spec)
split_dir <- file.path(project_root, "data", "processed", "splits")
dir.create(split_dir, showWarnings = FALSE, recursive = TRUE)

saveRDS(train_df, file.path(split_dir, "train_rows_protocol.rds"))
saveRDS(val_df,   file.path(split_dir, "val_rows_protocol.rds"))
saveRDS(test_df,  file.path(split_dir, "test_rows_protocol.rds"))
saveRDS(loso_folds, file.path(split_dir, "loso_folds_train_protocol.rds"))

manifest <- tibble::tibble(
  role = c(rep("train", length(train_subjects)), "validation", "test",
           rep("excluded_small", nrow(excluded_subjects))),
  subject_id = c(train_subjects, val_subject, test_subject, excluded_subjects$subject_id),
  rows = c(rows_by_subject$rows[match(train_subjects, rows_by_subject$subject_id)],
           rows_by_subject$rows[match(val_subject, rows_by_subject$subject_id)],
           rows_by_subject$rows[match(test_subject, rows_by_subject$subject_id)],
           excluded_subjects$rows)
) %>%
  arrange(factor(role, levels = c("train","validation","test","excluded_small")), desc(rows))

manifest_path <- file.path(split_dir, "split_manifest_protocol.csv")
write_csv_safe2(manifest, manifest_path)
log_table_preview(manifest, n = 50, title = "Split manifest (subject-level):")

# Save a compact split spec for reporting
split_spec <- list(
  protocol_only = TRUE,
  min_rows_per_subject = MIN_ROWS_PER_SUBJECT,
  eligible_subjects = eligible_subject_ids,
  excluded_subjects = excluded_subjects,
  train_subjects = train_subjects,
  validation_subject = val_subject,
  test_subject = test_subject,
  loso_folds_train = loso_folds
)
saveRDS(split_spec, file.path(split_dir, "split_spec_protocol.rds"))
log_info(paste0("Saved split spec: ", file.path(split_dir, "split_spec_protocol.rds")))

############################
# 33) Post-clean EDA report snippet (markdown)
############################
md_path_post <- file.path(post_out_dir, "post_clean_summary.md")
md_lines_post <- c(
  "# Post-clean Validation Summary (Protocol Only)",
  "",
  "## Validation Checks",
  "- Protocol-only confirmed (no optional sessions).",
  "- activity_id == 0 removed.",
  "- Orientation columns removed (dataset marks them invalid).",
  "- acc6 columns " %+% ifelse(length(present_acc6) == 0, "removed.", "present (kept intentionally)."),
  "",
  "## Subject-independent Split",
  sprintf("- Eligible subjects (min rows %d): %s", MIN_ROWS_PER_SUBJECT, paste(eligible_subject_ids, collapse = ", ")),
  sprintf("- Train subjects: %s", paste(train_subjects, collapse = ", ")),
  sprintf("- Validation subject: %s", val_subject),
  sprintf("- Test subject: %s", test_subject),
  sprintf("- Train rows: %s | Val rows: %s | Test rows: %s",
          scales::comma(nrow(train_df)), scales::comma(nrow(val_df)), scales::comma(nrow(test_df))),
  "",
  "## Artifacts",
  "- post_clean_global_summary.csv",
  "- post_clean_missingness_by_feature.csv + figure",
  "- post_clean_activity_distribution_overall.csv + figure",
  "- post_clean_rows_by_subject.csv + figure",
  "- post_clean_activity_coverage_by_subject.csv + figure",
  "- split_manifest_protocol.csv",
  "- loso_folds_train_protocol.rds"
)
writeLines(md_lines_post, md_path_post)
log_info(paste0("Saved: ", md_path_post))

log_step("STEP 04 completed successfully")


##############################################
# STEP 05 WINDOWING + FEATURE EXTRACTION (Protocol only, no leakage)
##############################################
log_step("STEP 05 WINDOWING + FEATURE EXTRACTION (Protocol only, no leakage)")

############################
# 34) Load row-level splits
############################
split_dir <- file.path(project_root, "data", "processed", "splits")
train_path <- file.path(split_dir, "train_rows_protocol.rds")
val_path   <- file.path(split_dir, "val_rows_protocol.rds")
test_path  <- file.path(split_dir, "test_rows_protocol.rds")

if (!file.exists(train_path) || !file.exists(val_path) || !file.exists(test_path)) {
  stop_with_context("Row-level split files not found. Run STEP 04 first.")
}

train_rows <- readRDS(train_path)
val_rows   <- readRDS(val_path)
test_rows  <- readRDS(test_path)

log_info(sprintf("Loaded row-level splits: train=%s | val=%s | test=%s rows",
                 scales::comma(nrow(train_rows)), scales::comma(nrow(val_rows)), scales::comma(nrow(test_rows))))

############################
# 35) Guardrails: same activity set across splits (or handle it explicitly)
############################
act_train <- sort(unique(as.integer(train_rows$activity_id)))
act_val   <- sort(unique(as.integer(val_rows$activity_id)))
act_test  <- sort(unique(as.integer(test_rows$activity_id)))

missing_in_train <- setdiff(union(act_val, act_test), act_train)
if (length(missing_in_train) > 0) {
  log_warn(paste0("WARNING: Some activities exist in val/test but NOT in train: ", paste(missing_in_train, collapse = ", ")))
  log_warn("This can break modeling or inflate/deflate metrics. Options:")
  log_warn(" - Swap which subject is val/test, OR")
  log_warn(" - Drop those activities from ALL splits (closed-set classification).")
  
  # Conservative default: drop those activities from all splits to keep a closed-set problem
  DROP_MISSING_TRAIN_ACTIVITIES <- TRUE
  
  if (DROP_MISSING_TRAIN_ACTIVITIES) {
    log_warn("Applying closed-set enforcement: dropping activities missing from train in ALL splits.")
    keep_ids <- act_train
    train_rows <- train_rows %>% dplyr::filter(activity_id %in% keep_ids)
    val_rows   <- val_rows   %>% dplyr::filter(activity_id %in% keep_ids)
    test_rows  <- test_rows  %>% dplyr::filter(activity_id %in% keep_ids)
    
    log_info(sprintf("After closed-set filter: train=%s | val=%s | test=%s rows",
                     scales::comma(nrow(train_rows)), scales::comma(nrow(val_rows)), scales::comma(nrow(test_rows))))
  }
}

############################
# 36) Windowing parameters
############################
# Typical HAR settings; tuned later if needed.
WINDOW_SECONDS <- 5
STEP_SECONDS   <- 1

# Require enough samples per window (100 Hz nominal -> ~500 rows in 5s).
# Use a conservative lower bound to tolerate slight irregularities.
MIN_ROWS_PER_WINDOW <- 250

# Label purity threshold: keep mostly single-activity windows (avoid transitions).
LABEL_PURITY_MIN <- 0.90

log_info("Windowing parameters:")
log_info(paste0(" - WINDOW_SECONDS:       ", WINDOW_SECONDS))
log_info(paste0(" - STEP_SECONDS:         ", STEP_SECONDS))
log_info(paste0(" - MIN_ROWS_PER_WINDOW:  ", MIN_ROWS_PER_WINDOW))
log_info(paste0(" - LABEL_PURITY_MIN:     ", LABEL_PURITY_MIN))

############################
# 37) Feature columns selection (numeric sensors only)
############################
id_cols <- c("timestamp","activity_id","activity","subject_id","source_file","session_type","row_in_file")

# keep only numeric sensor channels
sensor_cols <- setdiff(names(train_rows), id_cols)
sensor_cols <- sensor_cols[vapply(train_rows[sensor_cols], is.numeric, logical(1))]

if (length(sensor_cols) == 0) stop_with_context("No numeric sensor columns detected for feature extraction.")

log_info(sprintf("Detected %d numeric sensor columns for features.", length(sensor_cols)))
log_info(paste0("Example sensor columns: ", paste(head(sensor_cols, 10), collapse = ", "), ifelse(length(sensor_cols) > 10, " ...", "")))

############################
# 38) Robust feature functions (fast + dependency-light)
############################
calc_rms <- function(x) sqrt(mean(x^2, na.rm = TRUE))

calc_iqr <- function(x) stats::IQR(x, na.rm = TRUE)

# returns a named list of aggregated stats for one numeric vector
feat_stats <- function(x) {
  list(
    mean = mean(x, na.rm = TRUE),
    sd   = stats::sd(x, na.rm = TRUE),
    min  = min(x, na.rm = TRUE),
    max  = max(x, na.rm = TRUE),
    med  = stats::median(x, na.rm = TRUE),
    iqr  = calc_iqr(x),
    rms  = calc_rms(x)
  )
}

############################
# 39) Window builder using foverlaps (timestamp-based, no crossing file boundaries)
############################
make_windows_for_one_file <- function(df_one_file) {
  # df_one_file: data.frame for a single (subject_id, source_file), already Protocol-only and cleaned
  DT <- data.table::as.data.table(df_one_file)
  data.table::setorder(DT, timestamp)
  
  # create window grid
  ts_min <- DT[, min(timestamp, na.rm = TRUE)]
  ts_max <- DT[, max(timestamp, na.rm = TRUE)]
  
  if (!is.finite(ts_min) || !is.finite(ts_max) || ts_max <= ts_min) return(data.table::data.table())
  
  starts <- seq(from = ts_min, to = ts_max - WINDOW_SECONDS, by = STEP_SECONDS)
  if (length(starts) < 1) return(data.table::data.table())
  
  windows <- data.table::data.table(
    window_id = seq_along(starts),
    w_start = starts,
    w_end   = starts + WINDOW_SECONDS
  )
  data.table::setnames(windows, c("w_start","w_end"), c("start","end"))
  data.table::setkey(windows, start, end)
  
  # points as intervals [timestamp, timestamp]
  points <- DT[, c("timestamp","activity_id","activity","subject_id","source_file", sensor_cols), with = FALSE]
  points[, start := timestamp]
  points[, end   := timestamp]
  data.table::setkey(points, start, end)
  
  joined <- data.table::foverlaps(points, windows, by.x = c("start","end"), by.y = c("start","end"), type = "within", nomatch = 0L)
  if (nrow(joined) == 0) return(data.table::data.table())
  
  # label purity per window
  # counts per activity_id in each window
  lab_counts <- joined[, .N, by = .(window_id, activity_id)]
  lab_top <- lab_counts[, .(top_n = max(N), top_activity_id = activity_id[which.max(N)]), by = window_id]
  
  win_n <- joined[, .(n_rows = .N,
                      subject_id = subject_id[1],
                      source_file = source_file[1],
                      window_start = min(timestamp),
                      window_end = max(timestamp)),
                  by = window_id]
  
  win_meta <- merge(win_n, lab_top, by = "window_id", all.x = TRUE)
  win_meta[, label_purity := top_n / n_rows]
  
  # filter by rows and purity
  keep <- win_meta[n_rows >= MIN_ROWS_PER_WINDOW & label_purity >= LABEL_PURITY_MIN, window_id]
  if (length(keep) == 0) return(data.table::data.table())
  
  joined_kept <- joined[window_id %in% keep]
  
  # aggregate features per window over sensor columns
  # produce columns like: hand_acc16_x_mean, hand_acc16_x_sd, ...
  feats <- joined_kept[, {
    out <- list()
    for (cn in sensor_cols) {
      st <- feat_stats(get(cn))
      out[[paste0(cn, "_mean")]] <- st$mean
      out[[paste0(cn, "_sd")]]   <- st$sd
      out[[paste0(cn, "_min")]]  <- st$min
      out[[paste0(cn, "_max")]]  <- st$max
      out[[paste0(cn, "_med")]]  <- st$med
      out[[paste0(cn, "_iqr")]]  <- st$iqr
      out[[paste0(cn, "_rms")]]  <- st$rms
    }
    out
  }, by = window_id]
  
  # attach metadata/labels
  final <- merge(win_meta[window_id %in% keep], feats, by = "window_id", all.x = TRUE)
  
  # map label activity name consistently (from original df)
  # safest: take first matching activity string in the file
  act_map <- unique(DT[, .(activity_id, activity)])
  final <- merge(final, act_map, by.x = "top_activity_id", by.y = "activity_id", all.x = TRUE, suffixes = c("", "_label"))
  data.table::setnames(final, "activity", "label_activity")
  
  # final label id
  final[, label_activity_id := top_activity_id]
  final[, top_activity_id := NULL]
  
  final[]
}

build_window_dataset <- function(df_split, split_name) {
  log_step(paste0("Windowing split: ", split_name))
  
  # split by subject and file to avoid crossing boundaries
  idx <- df_split %>%
    dplyr::distinct(subject_id, source_file) %>%
    dplyr::arrange(subject_id, source_file)
  
  log_info(sprintf("%s: %d subject-file chunks", split_name, nrow(idx)))
  
  chunks <- vector("list", nrow(idx))
  for (i in seq_len(nrow(idx))) {
    sid <- idx$subject_id[i]
    sf  <- idx$source_file[i]
    
    df_one <- df_split %>% dplyr::filter(subject_id == sid, source_file == sf)
    
    log_info(sprintf("%s: subject=%s file=%s rows=%s",
                     split_name, sid, sf, scales::comma(nrow(df_one))))
    
    chunks[[i]] <- make_windows_for_one_file(df_one)
  }
  
  out <- data.table::rbindlist(chunks, fill = TRUE)
  out <- as.data.frame(out)
  
  if (nrow(out) == 0) {
    log_warn(paste0("No windows produced for split: ", split_name))
    return(out)
  }
  
  log_info(sprintf("%s: produced %s windows", split_name, scales::comma(nrow(out))))
  out
}

############################
# 40) Build window-level datasets
############################
train_win <- build_window_dataset(train_rows, "TRAIN")
val_win   <- build_window_dataset(val_rows,   "VALIDATION")
test_win  <- build_window_dataset(test_rows,  "TEST")

############################
# 41) Save window datasets + summaries
############################
win_dir <- file.path(project_root, "data", "processed", "windows")
dir.create(win_dir, showWarnings = FALSE, recursive = TRUE)

saveRDS(train_win, file.path(win_dir, "train_windows_protocol.rds"))
saveRDS(val_win,   file.path(win_dir, "val_windows_protocol.rds"))
saveRDS(test_win,  file.path(win_dir, "test_windows_protocol.rds"))

log_info("Saved window datasets:")
log_info(paste0(" - ", file.path(win_dir, "train_windows_protocol.rds")))
log_info(paste0(" - ", file.path(win_dir, "val_windows_protocol.rds")))
log_info(paste0(" - ", file.path(win_dir, "test_windows_protocol.rds")))

# Summaries by label
summarize_windows <- function(df, name) {
  if (nrow(df) == 0) return(tibble::tibble(split = name, label_activity_id = integer(), label_activity = character(), windows = integer(), pct = numeric()))
  s <- df %>%
    dplyr::count(label_activity_id, label_activity, name = "windows") %>%
    dplyr::mutate(pct = windows / sum(windows), split = name) %>%
    dplyr::arrange(dplyr::desc(windows))
  s
}

train_sum <- summarize_windows(train_win, "train")
val_sum   <- summarize_windows(val_win,   "validation")
test_sum  <- summarize_windows(test_win,  "test")

win_summary <- dplyr::bind_rows(train_sum, val_sum, test_sum)
win_summary_path <- file.path(project_root, "outputs", "post_clean_eda", "window_label_distribution.csv")
readr::write_csv(win_summary, win_summary_path)
log_info(paste0("Saved: ", win_summary_path))

log_table_preview(train_sum, n = 15, title = "TRAIN window label distribution (top 15):")
log_table_preview(val_sum,   n = 15, title = "VALIDATION window label distribution (top 15):")
log_table_preview(test_sum,  n = 15, title = "TEST window label distribution (top 15):")

log_step("STEP 05 completed successfully")

##############################################
# STEP 06 NORMALIZATION (TRAIN-only) + BASELINE MODELS (no leakage)
##############################################
log_step("STEP 06 NORMALIZATION (TRAIN-only) + BASELINE MODELS (no leakage)")

############################
# 42) Packages for modeling
############################
model_pkgs <- c("glmnet", "ranger", "Matrix", "caret")
install_if_missing(model_pkgs)
invisible(lapply(model_pkgs, library, character.only = TRUE))
set.seed(42)
log_info("Modeling packages loaded.")

############################
# 43) Load window-level datasets
############################
win_dir <- file.path(project_root, "data", "processed", "windows")
train_win_path <- file.path(win_dir, "train_windows_protocol.rds")
val_win_path   <- file.path(win_dir, "val_windows_protocol.rds")
test_win_path  <- file.path(win_dir, "test_windows_protocol.rds")

if (!file.exists(train_win_path) || !file.exists(val_win_path) || !file.exists(test_win_path)) {
  stop_with_context("Window datasets not found. Run STEP 05 first.")
}

train_win <- readRDS(train_win_path)
val_win   <- readRDS(val_win_path)
test_win  <- readRDS(test_win_path)

log_info(sprintf("Loaded windows: train=%s | val=%s | test=%s",
                 scales::comma(nrow(train_win)), scales::comma(nrow(val_win)), scales::comma(nrow(test_win))))

if (nrow(train_win) == 0 || nrow(val_win) == 0 || nrow(test_win) == 0) {
  stop_with_context("One or more window splits are empty. Adjust windowing params in STEP 05.")
}

############################
# 44) Define label + feature columns
############################
# label_activity is the class name; label_activity_id numeric.
# Keep a stable factor level set based on TRAIN only.
y_train_raw <- as.character(train_win$label_activity)
if (anyNA(y_train_raw)) stop_with_context("Missing labels in train windows.")

class_levels <- sort(unique(y_train_raw))
log_info(paste0("Classes (TRAIN): ", paste(class_levels, collapse = ", ")))

# Guardrail: ensure val/test labels are subset of train labels
val_extra  <- setdiff(unique(as.character(val_win$label_activity)), class_levels)
test_extra <- setdiff(unique(as.character(test_win$label_activity)), class_levels)
if (length(val_extra) > 0 || length(test_extra) > 0) {
  stop_with_context(
    paste0(
      "Label-set mismatch: found labels in val/test not present in train.\n",
      "val extra:  ", paste(val_extra, collapse = ", "), "\n",
      "test extra: ", paste(test_extra, collapse = ", "), "\n",
      "Fix by re-selecting subjects in STEP 04 or applying closed-set enforcement in STEP 05."
    )
  )
}

# Metadata columns to exclude from features
meta_cols <- c(
  "window_id","subject_id","source_file",
  "window_start","window_end",
  "n_rows","top_n","label_purity",
  "label_activity_id","label_activity"
)

# Feature columns = numeric non-meta columns
feature_cols <- setdiff(names(train_win), meta_cols)
feature_cols <- feature_cols[vapply(train_win[feature_cols], is.numeric, logical(1))]
if (length(feature_cols) == 0) stop_with_context("No numeric feature columns found in window dataset.")

# Align features across splits (intersection)
feature_cols <- Reduce(intersect, list(
  feature_cols,
  setdiff(names(val_win), meta_cols),
  setdiff(names(test_win), meta_cols)
))
feature_cols <- feature_cols[vapply(train_win[feature_cols], is.numeric, logical(1))]

log_info(sprintf("Feature columns: %d", length(feature_cols)))
log_info(paste0("Example features: ", paste(head(feature_cols, 10), collapse = ", "), ifelse(length(feature_cols) > 10, " ...", "")))

# Build X/Y
make_xy <- function(df) {
  x <- as.matrix(df[, feature_cols, drop = FALSE])
  y <- factor(as.character(df$label_activity), levels = class_levels)
  list(x = x, y = y)
}

xy_train <- make_xy(train_win)
xy_val   <- make_xy(val_win)
xy_test  <- make_xy(test_win)

############################
# 45) TRAIN-only normalization (center/scale), save scaler
############################
fit_scaler <- function(x_train) {
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, stats::sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(mu = mu, sd = sdv)
}

apply_scaler <- function(x, scaler) {
  sweep(sweep(x, 2, scaler$mu, "-"), 2, scaler$sd, "/")
}

scaler <- fit_scaler(xy_train$x)
x_train_s <- apply_scaler(xy_train$x, scaler)
x_val_s   <- apply_scaler(xy_val$x, scaler)
x_test_s  <- apply_scaler(xy_test$x, scaler)

baseline_out_dir <- file.path(project_root, "outputs", "baselines")
dir.create(baseline_out_dir, showWarnings = FALSE, recursive = TRUE)

scaler_path <- file.path(baseline_out_dir, "scaler_train_only.rds")
saveRDS(list(feature_cols = feature_cols, scaler = scaler), scaler_path)
log_info(paste0("Saved TRAIN-only scaler: ", scaler_path))

############################
# 46) Metrics utilities (Accuracy + Macro-F1 + Confusion)
############################
confusion_df <- function(y_true, y_pred, levels) {
  y_true <- factor(y_true, levels = levels)
  y_pred <- factor(y_pred, levels = levels)
  as.data.frame(table(truth = y_true, pred = y_pred))
}

macro_f1 <- function(y_true, y_pred, levels) {
  y_true <- factor(y_true, levels = levels)
  y_pred <- factor(y_pred, levels = levels)
  
  cm <- table(y_true, y_pred)
  
  f1s <- sapply(seq_along(levels), function(i) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    prec <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    if (is.na(prec) || is.na(rec) || (prec + rec) == 0) return(NA_real_)
    2 * prec * rec / (prec + rec)
  })
  
  mean(f1s, na.rm = TRUE)
}

eval_split <- function(name, y_true, y_pred, levels) {
  acc <- mean(y_true == y_pred)
  mf1 <- macro_f1(y_true, y_pred, levels)
  tibble::tibble(split = name, accuracy = acc, macro_f1 = mf1)
}

save_confusion_artifacts <- function(prefix, y_true, y_pred, levels) {
  dfc <- confusion_df(y_true, y_pred, levels)
  path_csv <- file.path(baseline_out_dir, paste0(prefix, "_confusion.csv"))
  readr::write_csv(dfc, path_csv)
  log_info(paste0("Saved: ", path_csv))
  
  # Heatmap figure
  p <- ggplot(dfc, aes(x = pred, y = truth, fill = Freq)) +
    geom_tile() +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(drop = FALSE) +
    labs(title = paste0("Confusion Matrix  ", prefix), x = "Predicted", y = "True") +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  fig <- file.path(baseline_out_dir, paste0(prefix, "_confusion.png"))
  ggplot2::ggsave(fig, p, width = 10, height = 8, dpi = 150)
  log_info(paste0("Saved figure: ", fig))
}

############################
# 47) Optional: Class weights (helps with imbalance like rope_jumping)
############################
USE_CLASS_WEIGHTS <- TRUE
if (USE_CLASS_WEIGHTS) {
  tab <- table(xy_train$y)
  class_weights <- as.numeric(sum(tab) / (length(tab) * tab))
  names(class_weights) <- names(tab)
  obs_weights <- class_weights[as.character(xy_train$y)]
  log_info("Using class weights (inverse frequency) for training.")
} else {
  class_weights <- NULL
  obs_weights <- rep(1, length(xy_train$y))
  log_info("Training without class weights.")
}

############################
# 48) Baseline Model A Multinomial Logistic Regression (glmnet)
############################
log_step("Baseline A glmnet multinomial (train-only scaling)")

x_train_sparse <- Matrix::Matrix(x_train_s, sparse = TRUE)
x_val_sparse   <- Matrix::Matrix(x_val_s, sparse = TRUE)
x_test_sparse  <- Matrix::Matrix(x_test_s, sparse = TRUE)

# glmnet expects y as factor for multinomial
cv_fit <- glmnet::cv.glmnet(
  x = x_train_sparse,
  y = xy_train$y,
  family = "multinomial",
  type.multinomial = "grouped",
  alpha = 0,                # ridge for stability (good baseline)
  weights = obs_weights,
  nfolds = 5
)
# Visualize the cross-validated error curve as a function of lambda.
# The first dotted vertical line (left) marks lambda.min (the value that
# minimizes CV error); the right dotted line marks lambda.1se.
plot(cv_fit)
title("Lambda optimization via cross-validation", line = 3)

best_lambda <- cv_fit$lambda.min
log_info(sprintf("glmnet selected lambda.min = %.6g", best_lambda))

glmnet_fit <- glmnet::glmnet(
  x = x_train_sparse,
  y = xy_train$y,
  family = "multinomial",
  type.multinomial = "grouped",
  alpha = 0,
  lambda = best_lambda,
  weights = obs_weights
)

glmnet_path <- file.path(baseline_out_dir, "model_glmnet_multinomial.rds")
saveRDS(list(model = glmnet_fit, lambda = best_lambda, levels = class_levels, feature_cols = feature_cols, scaler = scaler),
        glmnet_path)
log_info(paste0("Saved: ", glmnet_path))

pred_glmnet <- function(model, xmat) {
  pr <- predict(model, newx = xmat, type = "class")
  as.character(pr[,1])
}

train_pred_glm <- pred_glmnet(glmnet_fit, x_train_sparse)
val_pred_glm   <- pred_glmnet(glmnet_fit, x_val_sparse)
test_pred_glm  <- pred_glmnet(glmnet_fit, x_test_sparse)

m_glm_train <- eval_split("train", as.character(xy_train$y), train_pred_glm, class_levels) %>% mutate(model = "glmnet_ridge")
m_glm_val   <- eval_split("validation", as.character(xy_val$y),   val_pred_glm,   class_levels) %>% mutate(model = "glmnet_ridge")
m_glm_test  <- eval_split("test", as.character(xy_test$y),  test_pred_glm,  class_levels) %>% mutate(model = "glmnet_ridge")

save_confusion_artifacts("glmnet_train", as.character(xy_train$y), train_pred_glm, class_levels)
save_confusion_artifacts("glmnet_validation", as.character(xy_val$y), val_pred_glm, class_levels)
save_confusion_artifacts("glmnet_test", as.character(xy_test$y), test_pred_glm, class_levels)

############################
# 49) Baseline Model B Random Forest (ranger)
############################
log_step("Baseline B ranger random forest (no scaling needed, but we use same scaled matrix for consistency)")

# Build data.frames for ranger (it likes data.frames)
train_rf <- as.data.frame(x_train_s)
train_rf$y <- xy_train$y

val_rf <- as.data.frame(x_val_s)
test_rf <- as.data.frame(x_test_s)

rf_fit <- ranger::ranger(
  formula = y ~ .,
  data = train_rf,
  num.trees = 300,
  mtry = max(1, floor(sqrt(ncol(train_rf) - 1))),
  min.node.size = 5,
  probability = FALSE,
  class.weights = if (USE_CLASS_WEIGHTS) class_weights else NULL,
  seed = 42
)

rf_path <- file.path(baseline_out_dir, "model_ranger_rf.rds")
saveRDS(list(model = rf_fit, levels = class_levels, feature_cols = feature_cols, scaler = scaler),
        rf_path)
log_info(paste0("Saved: ", rf_path))

train_pred_rf <- predict(rf_fit, data = train_rf)$predictions
val_pred_rf   <- predict(rf_fit, data = val_rf)$predictions
test_pred_rf  <- predict(rf_fit, data = test_rf)$predictions

m_rf_train <- eval_split("train", as.character(xy_train$y), as.character(train_pred_rf), class_levels) %>% mutate(model = "ranger_rf")
m_rf_val   <- eval_split("validation", as.character(xy_val$y),   as.character(val_pred_rf),   class_levels) %>% mutate(model = "ranger_rf")
m_rf_test  <- eval_split("test", as.character(xy_test$y),  as.character(test_pred_rf),  class_levels) %>% mutate(model = "ranger_rf")

save_confusion_artifacts("ranger_train", as.character(xy_train$y), as.character(train_pred_rf), class_levels)
save_confusion_artifacts("ranger_validation", as.character(xy_val$y), as.character(val_pred_rf), class_levels)
save_confusion_artifacts("ranger_test", as.character(xy_test$y), as.character(test_pred_rf), class_levels)

############################
# 50) Compare baselines on VALIDATION, pick best, then final TEST report
############################
metrics_all <- dplyr::bind_rows(
  m_glm_train, m_glm_val, m_glm_test,
  m_rf_train,  m_rf_val,  m_rf_test
) %>% arrange(model, split)

metrics_path <- file.path(baseline_out_dir, "baseline_metrics.csv")
readr::write_csv(metrics_all, metrics_path)
log_info(paste0("Saved: ", metrics_path))

log_info("Baseline metrics summary:")
print(metrics_all)

best_on_val <- metrics_all %>%
  filter(split == "validation") %>%
  arrange(desc(macro_f1), desc(accuracy)) %>%
  slice(1)

log_info("Best baseline on validation:")
print(best_on_val)

# Save a short markdown summary for report integration
md_baseline <- file.path(baseline_out_dir, "baseline_summary.md")
md_lines <- c(
  "# Baseline Models Summary (Protocol Only, No Leakage)",
  "",
  "## Windowing",
  sprintf("- Train windows: %s", scales::comma(nrow(train_win))),
  sprintf("- Validation windows: %s", scales::comma(nrow(val_win))),
  sprintf("- Test windows: %s", scales::comma(nrow(test_win))),
  sprintf("- Classes: %s", paste(class_levels, collapse = ", ")),
  "",
  "## Models",
  "- glmnet multinomial (ridge) with train-only scaling",
  "- ranger random forest with optional class weights",
  "",
  "## Selection Criterion",
  "- Pick best model by validation Macro-F1 (tie-break by accuracy).",
  "",
  "## Best on Validation",
  paste0("- Model: ", best_on_val$model),
  paste0("- Accuracy: ", sprintf("%.4f", best_on_val$accuracy)),
  paste0("- Macro-F1: ", sprintf("%.4f", best_on_val$macro_f1)),
  "",
  "## Artifacts",
  "- baseline_metrics.csv",
  "- confusion matrices (csv + png) for each model/split",
  "- scaler_train_only.rds"
)
writeLines(md_lines, md_baseline)
log_info(paste0("Saved: ", md_baseline))

log_step("STEP 06 completed successfully")

##############################################
# STEP 07 SUBJECT-AWARE TUNING (LOSO in TRAIN) + FINAL MODEL SELECTION (VAL) + ONE-SHOT TEST
##############################################
log_step("STEP 07 SUBJECT-AWARE TUNING (LOSO in TRAIN) + FINAL SELECTION (VAL) + ONE-SHOT TEST")

############################
# 51) Load windows + split spec (LOSO subjects)
############################
win_dir <- file.path(project_root, "data", "processed", "windows")
train_win <- readRDS(file.path(win_dir, "train_windows_protocol.rds"))
val_win   <- readRDS(file.path(win_dir, "val_windows_protocol.rds"))
test_win  <- readRDS(file.path(win_dir, "test_windows_protocol.rds"))

split_dir <- file.path(project_root, "data", "processed", "splits")
split_spec <- readRDS(file.path(split_dir, "split_spec_protocol.rds"))

train_subjects <- split_spec$train_subjects
val_subject    <- split_spec$validation_subject
test_subject   <- split_spec$test_subject

log_info(paste0("Train subjects: ", paste(train_subjects, collapse = ", ")))
log_info(paste0("Validation subject: ", val_subject))
log_info(paste0("Test subject: ", test_subject))

############################
# 52) Rebuild X/Y with train-only scaler (avoid leakage)
############################
baseline_out_dir <- file.path(project_root, "outputs", "baselines")
scaler_obj <- readRDS(file.path(baseline_out_dir, "scaler_train_only.rds"))
feature_cols <- scaler_obj$feature_cols
scaler <- scaler_obj$scaler

# label levels from TRAIN
class_levels <- sort(unique(as.character(train_win$label_activity)))

make_xy <- function(df) {
  x <- as.matrix(df[, feature_cols, drop = FALSE])
  y <- factor(as.character(df$label_activity), levels = class_levels)
  list(x = x, y = y)
}

fit_scaler <- function(x_train) {
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, stats::sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(mu = mu, sd = sdv)
}

apply_scaler <- function(x, scaler) {
  sweep(sweep(x, 2, scaler$mu, "-"), 2, scaler$sd, "/")
}

xy_train <- make_xy(train_win)
xy_val   <- make_xy(val_win)
xy_test  <- make_xy(test_win)

# Apply TRAIN-only scaler saved in STEP 06
x_train_s <- apply_scaler(xy_train$x, scaler)
x_val_s   <- apply_scaler(xy_val$x, scaler)
x_test_s  <- apply_scaler(xy_test$x, scaler)

# Utilities (reuse from STEP 06 if present; re-declare for safety)
macro_f1 <- function(y_true, y_pred, levels) {
  y_true <- factor(y_true, levels = levels)
  y_pred <- factor(y_pred, levels = levels)
  cm <- table(y_true, y_pred)
  
  f1s <- sapply(seq_along(levels), function(i) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    prec <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    if (is.na(prec) || is.na(rec) || (prec + rec) == 0) return(NA_real_)
    2 * prec * rec / (prec + rec)
  })
  mean(f1s, na.rm = TRUE)
}

eval_metrics <- function(split_name, y_true, y_pred, levels, model_name) {
  tibble::tibble(
    model = model_name,
    split = split_name,
    accuracy = mean(y_true == y_pred),
    macro_f1 = macro_f1(y_true, y_pred, levels)
  )
}

############################
# 53) Subject-aware foldid for glmnet (LOSO within TRAIN)
############################
# foldid assigns each TRAIN observation to a fold = subject_id
train_subject_id_vec <- as.integer(train_win$subject_id)
fold_subjects <- sort(unique(train_subject_id_vec))

# Guardrail: ensure folds correspond to the intended train_subjects
if (!all(sort(unique(train_subject_id_vec)) %in% sort(train_subjects))) {
  log_warn("Train window subject_ids do not perfectly match split_spec train_subjects. Proceeding with actual subjects found in train_win.")
}

foldid <- match(train_subject_id_vec, fold_subjects)  # 1..K
K <- length(fold_subjects)
log_info(paste0("LOSO folds in TRAIN: K=", K, " (one fold per subject)"))

############################
# 54) Tuning output folder
############################
tune_out_dir <- file.path(project_root, "outputs", "tuning")
dir.create(tune_out_dir, showWarnings = FALSE, recursive = TRUE)

############################
# 55) Class weights (optional, based on TRAIN)
############################
USE_CLASS_WEIGHTS <- TRUE
if (USE_CLASS_WEIGHTS) {
  tab <- table(xy_train$y)
  class_weights <- as.numeric(sum(tab) / (length(tab) * tab))
  names(class_weights) <- names(tab)
  obs_weights <- class_weights[as.character(xy_train$y)]
  log_info("Using class weights (inverse frequency) for tuning/training.")
} else {
  class_weights <- NULL
  obs_weights <- rep(1, length(xy_train$y))
  log_info("No class weights used.")
}

############################
# 56) Model 1 glmnet multinomial: tune alpha with subject-aware CV
############################
log_step("Tuning glmnet (multinomial) with LOSO foldid")

install_if_missing(c("glmnet", "Matrix"))
library(glmnet)
library(Matrix)

x_train_sparse <- Matrix::Matrix(x_train_s, sparse = TRUE)

ALPHAS <- c(0, 0.25, 0.5, 1.0)  # ridge -> lasso spectrum
glmnet_tune <- vector("list", length(ALPHAS))

for (i in seq_along(ALPHAS)) {
  a <- ALPHAS[i]
  log_info(sprintf("glmnet CV (LOSO) alpha=%.2f", a))
  
  cvfit <- glmnet::cv.glmnet(
    x = x_train_sparse,
    y = xy_train$y,
    family = "multinomial",
    type.multinomial = "grouped",
    alpha = a,
    weights = obs_weights,
    foldid = foldid,           # SUBJECT-AWARE CV
    type.measure = "class"
  )
  
  best_lambda <- cvfit$lambda.min
  min_cvm <- min(cvfit$cvm, na.rm = TRUE)
  
  glmnet_tune[[i]] <- tibble::tibble(
    alpha = a,
    lambda = best_lambda,
    cv_misclass = min_cvm
  )
}

glmnet_tune_df <- dplyr::bind_rows(glmnet_tune) %>% arrange(cv_misclass, alpha)
readr::write_csv(glmnet_tune_df, file.path(tune_out_dir, "glmnet_loso_tuning.csv"))
log_info("Saved: outputs/tuning/glmnet_loso_tuning.csv")
log_table_preview(glmnet_tune_df, n = 50, title = "glmnet LOSO tuning results:")

best_glm <- glmnet_tune_df %>% slice(1)
best_alpha <- best_glm$alpha
best_lambda <- best_glm$lambda
log_info(sprintf("Best glmnet by LOSO misclass: alpha=%.2f | lambda=%.6g", best_alpha, best_lambda))

# Train final glmnet on ALL TRAIN with chosen alpha/lambda
glmnet_fit_final <- glmnet::glmnet(
  x = x_train_sparse,
  y = xy_train$y,
  family = "multinomial",
  type.multinomial = "grouped",
  alpha = best_alpha,
  lambda = best_lambda,
  weights = obs_weights
)

pred_glmnet_class <- function(model, xmat_sparse) {
  pr <- predict(model, newx = xmat_sparse, type = "class")
  as.character(pr[, 1])
}

x_val_sparse  <- Matrix::Matrix(x_val_s, sparse = TRUE)
x_test_sparse <- Matrix::Matrix(x_test_s, sparse = TRUE)

glm_val_pred  <- pred_glmnet_class(glmnet_fit_final, x_val_sparse)
glm_test_pred <- pred_glmnet_class(glmnet_fit_final, x_test_sparse)

m_glm_val  <- eval_metrics("validation", as.character(xy_val$y),  glm_val_pred,  class_levels, "glmnet_tuned")
m_glm_test <- eval_metrics("test",       as.character(xy_test$y), glm_test_pred, class_levels, "glmnet_tuned")

log_info("Tuned glmnet metrics:")
print(dplyr::bind_rows(m_glm_val, m_glm_test))

saveRDS(
  list(model = glmnet_fit_final, alpha = best_alpha, lambda = best_lambda,
       feature_cols = feature_cols, scaler = scaler, levels = class_levels),
  file.path(tune_out_dir, "model_glmnet_tuned.rds")
)
log_info("Saved: outputs/tuning/model_glmnet_tuned.rds")

############################
# 57) Model 2 ranger RF: subject-aware tuning via LOSO loops
############################
log_step("Tuning ranger RF with LOSO (subject holdout inside TRAIN)")

install_if_missing(c("ranger"))
library(ranger)

# Prepare a data.frame once
train_rf_full <- as.data.frame(x_train_s)
train_rf_full$y <- xy_train$y
train_rf_full$subject_id <- train_win$subject_id  # keep for fold split

# Validation/test frames (no subject_id needed)
val_rf <- as.data.frame(x_val_s)
test_rf <- as.data.frame(x_test_s)

# Small, safe grid (expand later if needed)
p <- ncol(train_rf_full) - 2  # exclude y and subject_id
mtry_base <- max(1, floor(sqrt(p)))

RF_GRID <- tidyr::expand_grid(
  num.trees = c(200, 400),
  mtry = unique(pmax(1, c(mtry_base, floor(p/3)))),
  min.node.size = c(1, 5, 10)
)

log_info(sprintf("RF tuning grid size: %d", nrow(RF_GRID)))

rf_loso_eval_one <- function(num.trees, mtry, min.node.size) {
  # LOSO over subjects present in TRAIN
  subs <- sort(unique(train_rf_full$subject_id))
  fold_metrics <- vector("list", length(subs))
  
  for (i in seq_along(subs)) {
    hold <- subs[i]
    df_tr <- train_rf_full[train_rf_full$subject_id != hold, , drop = FALSE]
    df_te <- train_rf_full[train_rf_full$subject_id == hold, , drop = FALSE]
    
    df_tr$subject_id <- NULL
    df_te_subj <- df_te$subject_id
    df_te$subject_id <- NULL
    
    fit <- ranger::ranger(
      y ~ .,
      data = df_tr,
      num.trees = num.trees,
      mtry = mtry,
      min.node.size = min.node.size,
      probability = FALSE,
      class.weights = if (USE_CLASS_WEIGHTS) class_weights else NULL,
      seed = 42
    )
    
    pred <- predict(fit, data = df_te)$predictions
    fold_metrics[[i]] <- eval_metrics(
      split_name = paste0("loso_subj_", hold),
      y_true = as.character(df_te$y),
      y_pred = as.character(pred),
      levels = class_levels,
      model_name = "ranger_rf_loso"
    ) %>% dplyr::mutate(num.trees = num.trees, mtry = mtry, min.node.size = min.node.size, holdout_subject = hold)
  }
  
  dplyr::bind_rows(fold_metrics)
}

# Run grid
rf_results_list <- vector("list", nrow(RF_GRID))
for (g in seq_len(nrow(RF_GRID))) {
  params <- RF_GRID[g, ]
  log_info(sprintf("RF LOSO grid %d/%d: trees=%d mtry=%d min.node.size=%d",
                   g, nrow(RF_GRID), params$num.trees, params$mtry, params$min.node.size))
  
  rf_results_list[[g]] <- rf_loso_eval_one(params$num.trees, params$mtry, params$min.node.size)
}

rf_loso_raw <- dplyr::bind_rows(rf_results_list)
readr::write_csv(rf_loso_raw, file.path(tune_out_dir, "ranger_loso_raw.csv"))
log_info("Saved: outputs/tuning/ranger_loso_raw.csv")

rf_loso_summary <- rf_loso_raw %>%
  group_by(num.trees, mtry, min.node.size) %>%
  summarise(
    loso_accuracy = mean(accuracy, na.rm = TRUE),
    loso_macro_f1 = mean(macro_f1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(loso_macro_f1), desc(loso_accuracy))

readr::write_csv(rf_loso_summary, file.path(tune_out_dir, "ranger_loso_summary.csv"))
log_info("Saved: outputs/tuning/ranger_loso_summary.csv")
log_table_preview(rf_loso_summary, n = 50, title = "RF LOSO tuning summary (mean across subjects):")

best_rf <- rf_loso_summary %>% slice(1)
log_info("Best RF by LOSO macro-F1:")
print(best_rf)

# Train RF final on ALL TRAIN with best hyperparams
train_rf_final <- train_rf_full
train_rf_final$subject_id <- NULL

rf_fit_final <- ranger::ranger(
  y ~ .,
  data = train_rf_final,
  num.trees = best_rf$num.trees,
  mtry = best_rf$mtry,
  min.node.size = best_rf$min.node.size,
  probability = FALSE,
  class.weights = if (USE_CLASS_WEIGHTS) class_weights else NULL,
  seed = 42
)

rf_val_pred  <- predict(rf_fit_final, data = val_rf)$predictions
rf_test_pred <- predict(rf_fit_final, data = test_rf)$predictions

m_rf_val  <- eval_metrics("validation", as.character(xy_val$y),  as.character(rf_val_pred),  class_levels, "ranger_rf_tuned")
m_rf_test <- eval_metrics("test",       as.character(xy_test$y), as.character(rf_test_pred), class_levels, "ranger_rf_tuned")

log_info("Tuned RF metrics:")
print(dplyr::bind_rows(m_rf_val, m_rf_test))

saveRDS(
  list(model = rf_fit_final,
       params = best_rf,
       feature_cols = feature_cols, scaler = scaler, levels = class_levels),
  file.path(tune_out_dir, "model_ranger_rf_tuned.rds")
)
log_info("Saved: outputs/tuning/model_ranger_rf_tuned.rds")

############################
# 58) Select final model by VALIDATION (ONLY), then one-shot TEST report
############################
val_compare <- dplyr::bind_rows(m_glm_val, m_rf_val) %>%
  arrange(desc(macro_f1), desc(accuracy))

test_compare <- dplyr::bind_rows(m_glm_test, m_rf_test) %>%
  arrange(desc(macro_f1), desc(accuracy))

readr::write_csv(val_compare,  file.path(tune_out_dir, "validation_model_comparison.csv"))
readr::write_csv(test_compare, file.path(tune_out_dir, "test_model_comparison.csv"))

log_table_preview(val_compare,  n = 10, title = "Model comparison on VALIDATION (selection basis):")
log_table_preview(test_compare, n = 10, title = "Model comparison on TEST (one-shot report):")

best_by_val <- val_compare %>% slice(1)
log_info("FINAL model choice based on VALIDATION only:")
print(best_by_val)

# Save concise markdown summary
md_path <- file.path(tune_out_dir, "step07_tuning_summary.md")
md_lines <- c(
  "# Step 07 Subject-aware tuning summary",
  "",
  "## Tuning protocol",
  "- Inner evaluation: LOSO within TRAIN (one fold per subject in TRAIN).",
  "- Model selection: VALIDATION subject only (subject-independent).",
  "- Test set: reported once; not used for selection.",
  "",
  "## Best hyperparameters",
  sprintf("- glmnet: alpha=%.2f, lambda=%.6g", best_alpha, best_lambda),
  sprintf("- ranger: trees=%d, mtry=%d, min.node.size=%d",
          best_rf$num.trees, best_rf$mtry, best_rf$min.node.size),
  "",
  "## Validation comparison (selection basis)",
  sprintf("- Best model: %s", best_by_val$model),
  sprintf("- Validation accuracy: %.4f", best_by_val$accuracy),
  sprintf("- Validation macro-F1: %.4f", best_by_val$macro_f1),
  "",
  "## Artifacts",
  "- glmnet_loso_tuning.csv",
  "- ranger_loso_summary.csv (+ ranger_loso_raw.csv)",
  "- validation_model_comparison.csv",
  "- test_model_comparison.csv",
  "- model_glmnet_tuned.rds",
  "- model_ranger_rf_tuned.rds"
)
writeLines(md_lines, md_path)
log_info(paste0("Saved: ", md_path))

log_step("STEP 07 completed successfully")

##############################################
# STEP 08 FINAL TRAINING (TRAIN+VAL) + FINAL TEST EVALUATION + REPORT ARTIFACTS
##############################################
log_step("STEP 08 FINAL TRAINING (TRAIN+VAL) + FINAL TEST EVALUATION + REPORT ARTIFACTS")

############################
# 59) Load window datasets
############################
win_dir <- file.path(project_root, "data", "processed", "windows")
train_win <- readRDS(file.path(win_dir, "train_windows_protocol.rds"))
val_win   <- readRDS(file.path(win_dir, "val_windows_protocol.rds"))
test_win  <- readRDS(file.path(win_dir, "test_windows_protocol.rds"))

log_info(sprintf("Loaded windows: train=%s | val=%s | test=%s",
                 scales::comma(nrow(train_win)), scales::comma(nrow(val_win)), scales::comma(nrow(test_win))))

if (nrow(train_win) == 0 || nrow(val_win) == 0 || nrow(test_win) == 0) {
  stop_with_context("One or more window splits are empty. Cannot proceed.")
}

############################
# 60) Load tuned RF model parameters from STEP 07
############################
tune_out_dir <- file.path(project_root, "outputs", "tuning")
rf_tuned_path <- file.path(tune_out_dir, "model_ranger_rf_tuned.rds")
if (!file.exists(rf_tuned_path)) stop_with_context("Tuned RF model not found. Run STEP 07 first.")

rf_tuned_obj <- readRDS(rf_tuned_path)
rf_best_params <- rf_tuned_obj$params
log_info("Loaded tuned RF parameters:")
print(rf_best_params)

############################
# 61) Build TRAIN+VAL development set (no TEST)
############################
dev_win <- dplyr::bind_rows(train_win, val_win)
log_info(sprintf("Development set (train+val): %s windows", scales::comma(nrow(dev_win))))

# Label levels based on DEV (train+val)
class_levels <- sort(unique(as.character(dev_win$label_activity)))
log_info(paste0("Classes (DEV): ", paste(class_levels, collapse = ", ")))

# Guardrail: ensure TEST labels are subset of DEV labels
test_extra <- setdiff(unique(as.character(test_win$label_activity)), class_levels)
if (length(test_extra) > 0) {
  stop_with_context(paste0("Label mismatch: TEST has labels not present in DEV: ", paste(test_extra, collapse = ", ")))
}

############################
# 62) Feature columns (use same feature list as earlier if available)
############################
# Prefer using the feature_cols stored with tuned object (for strict reproducibility)
feature_cols <- rf_tuned_obj$feature_cols
if (is.null(feature_cols) || length(feature_cols) == 0) {
  # fallback: infer numeric feature columns excluding meta
  meta_cols <- c(
    "window_id","subject_id","source_file",
    "window_start","window_end",
    "n_rows","top_n","label_purity",
    "label_activity_id","label_activity"
  )
  feature_cols <- setdiff(names(dev_win), meta_cols)
  feature_cols <- feature_cols[vapply(dev_win[feature_cols], is.numeric, logical(1))]
}

if (length(feature_cols) == 0) stop_with_context("No feature columns detected for final training.")
log_info(sprintf("Final feature columns: %d", length(feature_cols)))

############################
# 63) Train-only vs DEV-only scaling decision
############################
# For RF, scaling is not required. I keep scaling for consistency and to reuse my existing pipeline.
# IMPORTANT: scaler must be fit WITHOUT TEST. Here I fit on DEV (train+val) only.
fit_scaler <- function(x_train) {
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, stats::sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(mu = mu, sd = sdv)
}
apply_scaler <- function(x, scaler) {
  sweep(sweep(x, 2, scaler$mu, "-"), 2, scaler$sd, "/")
}

x_dev <- as.matrix(dev_win[, feature_cols, drop = FALSE])
x_test <- as.matrix(test_win[, feature_cols, drop = FALSE])

dev_scaler <- fit_scaler(x_dev)
x_dev_s <- apply_scaler(x_dev, dev_scaler)
x_test_s <- apply_scaler(x_test, dev_scaler)

y_dev <- factor(as.character(dev_win$label_activity), levels = class_levels)
y_test <- factor(as.character(test_win$label_activity), levels = class_levels)

############################
# 64) Class weights (DEV)
############################
USE_CLASS_WEIGHTS <- TRUE
if (USE_CLASS_WEIGHTS) {
  tab <- table(y_dev)
  class_weights <- as.numeric(sum(tab) / (length(tab) * tab))
  names(class_weights) <- names(tab)
  log_info("Using class weights (inverse frequency) for final training.")
} else {
  class_weights <- NULL
  log_info("Final training without class weights.")
}

############################
# 65) Train FINAL RF model on DEV (train+val)
############################
install_if_missing(c("ranger"))
library(ranger)

dev_rf <- as.data.frame(x_dev_s)
dev_rf$y <- y_dev

test_rf <- as.data.frame(x_test_s)

# Importance: permutation is more meaningful but slower. Use impurity if runtime is an issue.
USE_PERMUTATION_IMPORTANCE <- TRUE
importance_mode <- if (USE_PERMUTATION_IMPORTANCE) "permutation" else "impurity"

log_info(paste0("Training FINAL RF on DEV with importance='", importance_mode, "'"))

rf_final <- ranger::ranger(
  formula = y ~ .,
  data = dev_rf,
  num.trees = rf_best_params$num.trees,
  mtry = rf_best_params$mtry,
  min.node.size = rf_best_params$min.node.size,
  probability = FALSE,
  class.weights = if (USE_CLASS_WEIGHTS) class_weights else NULL,
  importance = importance_mode,
  seed = 42
)

############################
# 66) Evaluate on TEST (one-shot reporting)
############################
pred_test <- predict(rf_final, data = test_rf)$predictions
pred_test <- factor(as.character(pred_test), levels = class_levels)

accuracy <- mean(pred_test == y_test)

macro_f1 <- function(y_true, y_pred, levels) {
  y_true <- factor(y_true, levels = levels)
  y_pred <- factor(y_pred, levels = levels)
  cm <- table(y_true, y_pred)
  
  f1s <- sapply(seq_along(levels), function(i) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    prec <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
    if (is.na(prec) || is.na(rec) || (prec + rec) == 0) return(NA_real_)
    2 * prec * rec / (prec + rec)
  })
  
  mean(f1s, na.rm = TRUE)
}

macroF1 <- macro_f1(y_test, pred_test, class_levels)

final_metrics <- tibble::tibble(
  model = "ranger_rf_final_dev",
  split = "test",
  accuracy = accuracy,
  macro_f1 = macroF1,
  num_trees = rf_best_params$num.trees,
  mtry = rf_best_params$mtry,
  min_node_size = rf_best_params$min.node.size,
  importance = importance_mode
)

log_info("FINAL TEST metrics (one-shot report):")
print(final_metrics)

############################
# 67) Per-class metrics + confusion matrix
############################
cm <- table(truth = y_test, pred = pred_test)

per_class <- tibble::tibble(
  class = class_levels,
  support = as.integer(rowSums(cm)),
  tp = as.integer(diag(cm)),
  fp = as.integer(colSums(cm) - diag(cm)),
  fn = as.integer(rowSums(cm) - diag(cm))
) %>%
  mutate(
    precision = dplyr::if_else(tp + fp == 0, NA_real_, tp / (tp + fp)),
    recall    = dplyr::if_else(tp + fn == 0, NA_real_, tp / (tp + fn)),
    f1        = dplyr::if_else(is.na(precision) | is.na(recall) | (precision + recall) == 0,
                               NA_real_, 2 * precision * recall / (precision + recall))
  ) %>%
  arrange(desc(f1))
print(per_class)
############################
# 68) Save report artifacts (CSV + figures + RDS)
############################
final_out_dir <- file.path(project_root, "outputs", "final")
dir.create(final_out_dir, showWarnings = FALSE, recursive = TRUE)

# Save model package
final_model_path <- file.path(final_out_dir, "final_model_ranger_rf_dev.rds")
saveRDS(
  list(
    model = rf_final,
    params = rf_best_params,
    feature_cols = feature_cols,
    scaler = dev_scaler,
    levels = class_levels,
    windowing = list(WINDOW_SECONDS = WINDOW_SECONDS, STEP_SECONDS = STEP_SECONDS,
                     MIN_ROWS_PER_WINDOW = MIN_ROWS_PER_WINDOW, LABEL_PURITY_MIN = LABEL_PURITY_MIN),
    split = list(train_subjects = split_spec$train_subjects,
                 validation_subject = split_spec$validation_subject,
                 test_subject = split_spec$test_subject)
  ),
  final_model_path
)
log_info(paste0("Saved final model object: ", final_model_path))

# CSVs
metrics_csv <- file.path(final_out_dir, "final_test_metrics.csv")
readr::write_csv(final_metrics, metrics_csv)
log_info(paste0("Saved: ", metrics_csv))

per_class_csv <- file.path(final_out_dir, "final_test_per_class_metrics.csv")
readr::write_csv(per_class, per_class_csv)
log_info(paste0("Saved: ", per_class_csv))

# Confusion matrix as long table
conf_long <- as.data.frame(cm)
conf_csv <- file.path(final_out_dir, "final_test_confusion.csv")
readr::write_csv(conf_long, conf_csv)
log_info(paste0("Saved: ", conf_csv))

# Confusion heatmap figure
p_conf <- ggplot(conf_long, aes(x = pred, y = truth, fill = Freq)) +
  geom_tile() +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(drop = FALSE) +
  labs(title = "Final Confusion Matrix TEST (Subject-independent)", x = "Predicted", y = "True") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

conf_png <- file.path(final_out_dir, "final_test_confusion.png")
ggplot2::ggsave(conf_png, p_conf, width = 10, height = 8, dpi = 150)
log_info(paste0("Saved figure: ", conf_png))
print(p_conf)
############################
# 69) Feature importance (top N)
############################
vi <- rf_final$variable.importance
if (!is.null(vi)) {
  vi_df <- tibble::tibble(
    feature = names(vi),
    importance = as.numeric(vi)
  ) %>%
    arrange(desc(importance))
  
  vi_csv <- file.path(final_out_dir, "final_feature_importance.csv")
  readr::write_csv(vi_df, vi_csv)
  log_info(paste0("Saved: ", vi_csv))
  
  TOP_N <- 30
  vi_top <- vi_df %>% slice_head(n = TOP_N) %>%
    mutate(feature = factor(feature, levels = rev(feature)))
  
  p_vi <- ggplot(vi_top, aes(x = feature, y = importance)) +
    geom_col() +
    coord_flip() +
    labs(title = paste0("Top ", TOP_N, " Feature Importances (", importance_mode, ")"),
         x = "Feature", y = "Importance") +
    theme_minimal(base_size = 12)
  
  vi_png <- file.path(final_out_dir, "final_feature_importance_top30.png")
  ggplot2::ggsave(vi_png, p_vi, width = 12, height = 9, dpi = 150)
  log_info(paste0("Saved figure: ", vi_png))
  
  log_table_preview(vi_df, n = 20, title = "Top 20 feature importances (final model):")
} else {
  log_warn("Variable importance is NULL. (This can happen if importance was disabled.)")
}

############################
# 70) Final markdown summary (ready to paste into report)
############################
final_md <- file.path(final_out_dir, "final_summary.md")
md_lines <- c(
  "# Final Model Summary (PAMAP2 Protocol Only)",
  "",
  "## Data & Split",
  "- Protocol files only; optional sessions excluded.",
  "- Subject-independent split:",
  paste0("  - Train subjects: ", paste(split_spec$train_subjects, collapse = ", ")),
  paste0("  - Validation subject: ", split_spec$validation_subject),
  paste0("  - Test subject: ", split_spec$test_subject),
  "",
  "## Windowing",
  paste0("- Window length (seconds): ", WINDOW_SECONDS),
  paste0("- Step / stride (seconds): ", STEP_SECONDS),
  paste0("- Min rows per window: ", MIN_ROWS_PER_WINDOW),
  paste0("- Label purity threshold: ", LABEL_PURITY_MIN),
  "",
  "## Final model",
  "- Model: Random Forest (ranger)",
  paste0("- num.trees: ", rf_best_params$num.trees),
  paste0("- mtry: ", rf_best_params$mtry),
  paste0("- min.node.size: ", rf_best_params$min.node.size),
  paste0("- class weights: ", ifelse(USE_CLASS_WEIGHTS, "YES (inverse frequency)", "NO")),
  paste0("- importance: ", importance_mode),
  "",
  "## Final TEST performance (one-shot report)",
  paste0("- Accuracy: ", sprintf("%.4f", accuracy)),
  paste0("- Macro-F1: ", sprintf("%.4f", macroF1)),
  "",
  "## Artifacts",
  "- final_model_ranger_rf_dev.rds",
  "- final_test_metrics.csv",
  "- final_test_per_class_metrics.csv",
  "- final_test_confusion.csv + final_test_confusion.png",
  "- final_feature_importance.csv + final_feature_importance_top30.png"
)
writeLines(md_lines, final_md)
log_info(paste0("Saved: ", final_md))
print(per_class)
log_step("STEP 08 completed successfully")
log_info("Next step: KNIT the RMD")

