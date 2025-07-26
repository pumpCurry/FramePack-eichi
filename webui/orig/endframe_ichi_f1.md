"""
CONFIG QUEUE SYSTEM OVERVIEW:
This system allows users to save UI configurations as JSON files and process
them automatically in sequence. Key components:

1. CONFIG FILE MANAGEMENT: Save/load UI states (image, prompt, LoRA settings)
2. QUEUE PROCESSING: Automatic sequential processing of multiple configs
3. UI INTEGRATION: Seamless integration with existing manual generation system
4. BATCH SUPPORT: Each config can generate multiple videos with seed incrementation

ARCHITECTURE:
- ConfigQueueManager: Handles file operations and queue state management
- UI Event Handlers: Bridge between Gradio UI and queue manager
- Processing Integration: Captures UI settings and applies to queue processing
- Status Monitoring: Real-time feedback and progress tracking

GLOBAL STATE VARIABLES:
- config_queue_manager: Main queue manager instance
- current_loaded_config: Currently loaded config name for UI
- queue_processing_active: Global flag for queue processing state
- current_processing_config_name: Config being processed (used for video naming)
- current_batch_progress: Tracks batch progress within current config
- queue_ui_settings: Captured UI settings for queue processing
"""

def get_current_ui_settings_for_queue():
    """
    CRITICAL FUNCTION: Captures comprehensive UI state for queue processing
    
    PURPOSE:
    When queue processing starts, this function takes a snapshot of ALL current
    UI settings and stores them globally. These settings are then applied to
    every config in the queue, ensuring consistent quality/duration parameters.
    
    ARCHITECTURE DECISION:
    - UI settings (quality/duration) come from current UI state
    - Config-specific settings (image/prompt/LoRA) come from individual config files
    - This separation allows batch processing with consistent technical parameters
    
    PARAMETERS CAPTURED:
    - Duration: total_second_length (prioritized over radio button)
    - Quality: steps, CFG, resolution, CRF
    - Generation: seed, random seed, TeaCache, FP8 optimization
    - System: GPU memory preservation
    - Output: save options, directories, alarms
    - F1 Mode: image strength, padding settings
    
    RETURNS:
    dict: Complete settings dictionary for queue processing
    
    MODIFICATION NOTES:
    - Add new UI parameters here when extending the system
    - Ensure type conversion safety (int/float/bool validation)
    - Update corresponding process() function parameters when adding settings
    """

def cancel_operation_handler():
    """
    CANCELLATION HANDLER: Handles cancellation of pending operations
    
    SIMPLE FUNCTION: Clears confirmation dialog and operation data without performing any actions.
    """

def merged_refresh_handler_standardized():
    """
    REFRESH HANDLER: Updates both config list and queue status with auto-correction
    
    AUTO-CORRECTION FEATURES:
    - Detects stuck queue states (processing flag set but no actual work)
    - Corrects inconsistent processing states between global and manager flags
    - Provides diagnostic logging for troubleshooting
    
    DUAL REFRESH:
    - Available configs: Fresh filesystem scan
    - Queue status: Current processing state with validation
    
    RETURNS:
    Tuple of (message, config_dropdown_update, queue_status_update)
    """

def queue_config_handler_with_confirmation(config_dropdown):
    """
    QUEUE HANDLER: Adds config to queue with overwrite confirmation
    
    CONFIRMATION SYSTEM:
    If config already exists in queue, shows confirmation dialog before overwriting.
    Otherwise, proceeds directly with queueing.
    
    RETURNS:
    Tuple of UI updates for message, status, confirmation, operation_data
    """

def stop_queue_processing_handler_fixed():
    """
    STOP HANDLER: Stops queue processing with proper state management
    
    STATE CLEANUP:
    - Resets both global and manager processing flags
    - Clears current config reference
    - Provides appropriate user feedback
    - Restores UI component visibility
    
    RETURNS:
    Tuple of (message, status_update)
    """

def clear_queue_handler():
    """
    CLEAR HANDLER: Removes all items from queue
    
    SAFETY: Only works when queue is not actively processing.
    
    RETURNS:
    Tuple of (message, status_update)
    """

def save_current_config_handler_v3(config_name_input, add_timestamp, input_image, prompt, use_lora, lora_mode, 
                                  lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files, 
                                  lora_files2, lora_files3, lora_scales_text):
    """
    PRIMARY SAVE HANDLER: Manages config saving with overwrite detection
    
    FLOW:
    1. Validate inputs (image, prompt)
    2. Check for existing config (if timestamp disabled)
    3. Show confirmation dialog for overwrites
    4. Store operation data for confirmation system
    5. Call perform_save_operation_v3() for actual saving
    
    RETURNS:
    Tuple of UI updates for message, dropdown, status, confirmation, operation_data
    """

def perform_save_operation_v3(config_name_input, add_timestamp, input_image, prompt, use_lora, lora_mode,
                            lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_files,
                            lora_files2, lora_files3, lora_scales_text):
    """
    ACTUAL SAVE OPERATION: Performs the config file saving
    
    CRITICAL FUNCTION: This is where actual config data is assembled and saved
    
    PROCESS:
    1. Assemble LoRA settings using get_current_lora_settings()
    2. Call ConfigQueueManager.save_config_with_timestamp_option()
    3. Parse returned config name (removing system messages)
    4. Update UI with new config and status
    
    CONFIG NAME PARSING:
    The function carefully extracts ONLY the config name from success messages,
    filtering out system messages like "(copied file)" to prevent UI contamination.
    """

def load_config_with_delayed_lora_application_fixed(config_name):
    """
    COMPLEX LOAD HANDLER: Loads config and applies LoRA settings immediately
    
    ADVANCED FEATURES:
    - Automatic image recovery from multiple locations
    - LoRA configuration restoration with validation
    - Language-independent config format handling
    - Graceful handling of missing files
    
    IMAGE RECOVERY PROCESS:
    1. Try original image path
    2. Try alternative paths stored in config
    3. Search by original filename in config_images/
    4. Allow loading even if image missing (for editing)
    
    LORA RESTORATION:
    - Handles both old and new config formats
    - Validates file existence before applying
    - Uses safe fallbacks for missing files
    
    RETURNS:
    List of UI updates for all relevant components
    """

def delete_config_handler_v2(config_dropdown):
    """
    DELETE HANDLER: Deletes config with confirmation system
    
    CONFIRMATION SYSTEM:
    Shows confirmation dialog before deletion to prevent accidental loss.
    
    VALIDATION:
    Checks file existence and refreshes list if file not found.
    
    RETURNS:
    Tuple of UI updates for message, dropdown, status, confirmation, operation_data
    """

def validate_and_process_with_queue_check(*args):
    """
    INTEGRATION BRIDGE: Modified validate_and_process with queue processing check
    
    MUTUAL EXCLUSION:
    Prevents manual generation when queue is processing, and vice versa.
    This ensures GPU resources are not conflicted between manual and queue processing.
    
    UI STATE MANAGEMENT:
    - Updates button states based on processing status
    - Provides clear feedback about why generation is blocked
    - Maintains consistent UI behavior
    
    YIELDS:
    Generator of UI updates matching manual generation output format
    """

def end_process_enhanced():
    """
    ENHANCED END HANDLER: Stops manual process with button state management
    
    BUTTON STATE COORDINATION:
    Updates both manual and queue button states appropriately when stopping manual generation.
    
    RETURNS:
    Tuple of button state updates
    """

def create_enhanced_config_queue_ui():
    """
    UI BUILDER: Creates complete config queue UI with all components
    
    ARCHITECTURE:
    - Config management section (save, load, delete)
    - Queue control section (queue, start, stop, clear)
    - Status display with real-time updates
    - Confirmation system for destructive operations
    
    RETURNS:
    Dictionary of all UI components for event handler registration
    """

def setup_enhanced_config_queue_events(components, ui_components):
    """
    EVENT REGISTRATION: Sets up all event handlers for config queue system
    
    COMPREHENSIVE EVENT HANDLING:
    - Config management events (save, load, delete)
    - Queue control events (start, stop, clear, queue)
    - Confirmation system events (confirm, cancel)
    - Refresh and status update events
    
    PARAMETERS:
    components: Dict of config queue UI components
    ui_components: Dict of main UI components for integration
    
    MODIFICATION NOTES:
    - Add new event handlers here when extending functionality
    - Maintain consistent parameter passing between handlers
    - Ensure proper error handling in all event handlers
    """

def setup_periodic_queue_status_check():
    """
    MONITORING SYSTEM: Periodic status checking with automatic state correction
    
    PURPOSE:
    Runs background thread to monitor queue state and automatically correct
    inconsistencies that may occur due to errors or unexpected conditions.
    
    AUTO-CORRECTION FEATURES:
    - Detects stuck processing states
    - Resets inconsistent flags
    - Logs corrections for debugging
    
    THREAD SAFETY:
    Uses daemon thread to avoid blocking application shutdown.
    """

def get_lora_mode_text(lora_mode_key):
    """
    UTILITY: Convert language-independent key to current localized text
    
    LANGUAGE INDEPENDENCE:
    Converts internal storage keys to current UI language for proper display.
    
    PARAMETERS:
    lora_mode_key: Language-independent key (LORA_MODE_DIRECTORY, LORA_MODE_UPLOAD, etc.)
    
    RETURNS:
    Localized text for current language
    """

def get_current_lora_settings(use_lora, lora_mode, lora_dropdown1, lora_dropdown2, lora_dropdown3, 
                             lora_files, lora_files2, lora_files3, lora_scales_text):
    """
    ADVANCED FUNCTION: LoRA Configuration Processing for Config Files
    
    PURPOSE:
    Processes LoRA settings from UI and converts them to a standardized format
    for config file storage. Handles both directory selection and file upload modes.
    
    ARCHITECTURE DECISION:
    - All config files use directory mode internally (language-independent)
    - File upload mode auto-converts to directory mode upon saving
    - Uses language-independent keys to avoid localization issues
    
    AUTO-CONVERSION PROCESS (File Upload → Directory Mode):
    1. Detect uploaded files
    2. Copy files to lora/ directory with deduplication
    3. Store as directory mode format in config
    4. Log conversion for user feedback
    
    RETURNS:
    dict: Standardized LoRA settings for config storage
    
    MODIFICATION NOTES:
    - When adding new LoRA parameters, update this function
    - Maintain language independence in stored keys
    - Handle file system operations safely (existence checks, permissions)
    """

def apply_lora_config_to_dropdowns_safe(lora_files, existing_choices=None):
    """
    UTILITY: Apply LoRA file configuration to dropdowns with validation
    
    SAFETY FEATURES:
    - Validates file existence before applying
    - Handles missing files gracefully
    - Ensures all dropdown values are valid choices
    - Provides detailed logging for troubleshooting
    
    PARAMETERS:
    lora_files: List of LoRA file paths
    existing_choices: Pre-scanned directory choices (optional)
    
    RETURNS:
    Tuple of (choices, dropdown_values, applied_files)
    """

def scan_lora_directory():
    """
    UTILITY: Scan ./lora directory for LoRA model files with enhanced validation
    
    FEATURES:
    - Creates directory if not exists
    - Validates file readability
    - Sorts results for consistent ordering
    - Ensures string type safety
    - Always includes "none" option
    
    RETURNS:
    List of valid LoRA filenames with "none" option first
    """

def start_queue_processing_with_current_ui_values(
    # Duration settings - both controls
    length_radio, total_second_length,
    # Frame settings
    frame_size_radio,
    # Quality settings  
    steps, cfg, gs, rs, resolution, mp4_crf,
    # Generation settings
    seed, use_random_seed, use_teacache, image_strength, fp8_optimization,
    # System settings
    gpu_memory_preservation,
    # Output settings
    keep_section_videos, save_section_frames, save_tensor_data, 
    frame_save_mode, output_dir, alarm_on_completion,
    # F1 mode settings
    all_padding_value, use_all_padding,
    # Batch count parameter
    batch_count
):
    """
        QUEUE PROCESSING ENTRY POINT: Initiates queue processing with UI settings capture
        
        CRITICAL ARCHITECTURE:
        This function is the bridge between the UI and queue processing system.
        It captures ALL current UI settings and stores them globally for queue processing.
        
        PROCESSING FLOW:
        1. Validate queue has items
        2. Capture UI settings snapshot (get_current_ui_settings_for_queue)
        3. Store settings globally for queue worker access
        4. Start queue processing thread
        5. Monitor progress with periodic status updates
        
        UI SETTINGS CAPTURE:
        - Quality/Duration settings from UI → Applied to ALL queue items
        - Image/Prompt/LoRA settings from configs → Per-config basis
        
        MONITORING SYSTEM:
        - Real-time status updates every 3 seconds
        - Batch progress tracking within each config
        - Automatic completion detection
        - Error handling and recovery
        
        YIELDS:
        Generator of UI updates for real-time feedback during processing
        
        MODIFICATION NOTES:
        - Add new UI parameters to function signature AND get_current_ui_settings_for_queue()
        - Maintain parameter order consistency with UI component arrangement
        - Update monitoring logic when adding new status information
        """

def process_config_item_with_batch_support(config_data):
    """
    ADVANCED QUEUE PROCESSOR: Processes individual config with full batch support
    
    ARCHITECTURE:
    This function is called by ConfigQueueManager for each queued config.
    It bridges between the queue system and the main video generation process.
    
    BATCH PROCESSING INTEGRATION:
    - Extracts batch_count from globally stored UI settings
    - Calls main process() function with config-specific parameters
    - Tracks batch progress for real-time UI updates
    - Handles seed incrementation per batch
    
    CONFIG DATA PROCESSING:
    - Image path validation and handling
    - LoRA settings reconstruction for both directory and upload modes
    - Prompt and other parameter extraction
    - Error handling for missing files
    
    VIDEO NAMING INTEGRATION:
    Sets global current_processing_config_name for video file naming in worker() function.
    
    RETURNS:
    bool: True if processing succeeded, False otherwise
    
    MODIFICATION NOTES:
    - When adding new config parameters, update config data extraction
    - Maintain batch progress tracking for UI feedback
    - Handle new parameter types in the process() function call
    """

def format_queue_status_with_batch_progress(status):
    """
    STATUS FORMATTER: Enhanced queue status display with batch progress
    
    PURPOSE:
    Formats queue status information for UI display, including:
    - Processing state and current config
    - Batch progress within current config
    - Queue count and pending items
    - Recently completed items (newest first)
    - Available configs count
    - Timestamp for updates
    
    BATCH PROGRESS INTEGRATION:
    Shows current batch within config processing for better user feedback.
    
    MODIFICATION NOTES:
    - Adjust CONST_queued_shown_count/CONST_latest_finish_count for display limits
    - Add new status fields as needed
    - Maintain readable formatting for different screen sizes
    """

def update_batch_progress(current_batch, total_batches):
    """
    BATCH PROGRESS TRACKER: Updates global batch progress state
    
    PURPOSE:
    Maintains real-time batch progress information for UI display during queue processing.
    
    GLOBAL STATE UPDATE:
    Updates current_batch_progress dictionary used by status formatting and monitoring.
    
    PARAMETERS:
    current_batch (int): Current batch number (0-based)
    total_batches (int): Total number of batches for current config
    """

def confirm_operation_handler_fixed(operation_data):
    """
    CONFIRMATION HANDLER: Processes confirmed operations with enhanced delete handling
    
    SUPPORTED OPERATIONS:
    - overwrite_exact: Config file overwriting
    - queue_overwrite: Queue item replacement
    - delete: Config file deletion (with name input clearing)
    
    DELETE ENHANCEMENT:
    Clears config name input field after successful deletion to prevent confusion.
    
    RETURNS:
    Tuple of UI updates for all relevant components
    """

def toggle_lora_full_update(use_lora_val):
    """
    ADVANCED LORA UI HANDLER: Enhanced LoRA toggle with config loading support
    
    PROBLEM SOLVED:
    Original toggle_lora_settings() function was inline in gr.Blocks and had issues:
    1. When loading configs with LoRA settings, UI wouldn't display correctly
    2. Pending LoRA config data wasn't being applied properly
    3. Dropdown choices weren't being refreshed when needed
    4. State inconsistencies between LoRA mode and dropdown values
    
    SOLUTION ARCHITECTURE:
    This enhanced function provides:
    1. PENDING CONFIG SUPPORT: Handles LoRA settings from loaded configs
    2. SMART STATE MANAGEMENT: Remembers previous LoRA mode when toggling
    3. FRESH DROPDOWN SCANNING: Ensures dropdown choices are current
    4. VALIDATION: Ensures all dropdown values are valid choices
    
    INTEGRATION WITH CONFIG SYSTEM:
    - Checks for pending_lora_config_data (set by config loading)
    - Reapplies stored LoRA settings when toggle is re-enabled
    - Maintains consistency between config loading and manual LoRA setup
    
    PARAMETERS:
    use_lora_val (bool): Whether LoRA is enabled
    
    RETURNS:
    List of Gradio updates for all LoRA-related components:
    [lora_mode, lora_upload_group, lora_dropdown_group, lora_scales_text,
     lora_dropdown1, lora_dropdown2, lora_dropdown3]
    
    WORKFLOW:
    1. Get basic visibility from original toggle_lora_settings()
    2. If LoRA disabled: Clear pending data, hide components
    3. If LoRA enabled: Check for pending config data
    4. If pending data exists: Reapply stored config settings
    5. If no pending data: Use previous mode or default to directory mode
    6. Refresh dropdown choices and validate values
    
    MODIFICATION NOTES:
    - Global variable 'pending_lora_config_data' stores config loading state
    - Global variable 'previous_lora_mode' remembers user's last choice
    - Always call scan_lora_directory() for fresh choices
    - Ensure all returned dropdown values exist in choices list
    """

    """
    DROPDOWN INITIALIZATION FIX: Ensures prompt preset dropdown has valid default value
    
    PROBLEM SOLVED:
    When the prompt preset dropdown was created, Gradio would sometimes:
    1. Have choices list but no valid default value selected
    2. Show empty dropdown even when presets were available
    3. Throw warnings about value not being in choices
    4. Default preset "起動時デフォルト" not being found in choices
    
    ROOT CAUSE:
    The preset system creates choices dynamically, and there was a timing issue
    where the dropdown was created before the choices were properly populated,
    or the default value wasn't included in the choices list.
    
    SOLUTION:
    This function ensures:
    1. Choices are properly loaded from preset system
    2. Default presets and user presets are separated and sorted
    3. "起動時デフォルト" (startup default) is always available
    4. Valid default value is returned for dropdown initialization
    
    USAGE:
    Called during UI setup to get proper choices and default value:
    ```python
    sorted_choices, default_value = fix_prompt_preset_dropdown_initialization()
    preset_dropdown = gr.Dropdown(
        choices=sorted_choices,
        value=default_value,  # Guaranteed to be valid
        ...
    )
    """