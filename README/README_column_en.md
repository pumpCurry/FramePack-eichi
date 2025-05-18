# FramePack-eichi Extended Documentation | [日本語](README_column.md) | [繁體中文](README_column_zh.md) | [Русский](README_column_ru.md)

This document serves as a detailed version of the FramePack-eichi main README, providing in-depth information about each feature and setting. For more practical usage methods, please refer to the [User Guide](README_userguide_en.md), and for update history, see the [Changelog](README_changelog.md).

## 🌟 Advanced Settings

### Model Selection (Added in v1.9.1)
- **F1 Model**: Forward generation method, more dynamic movement, simple operation
- **Standard Model**: Reverse generation method, precise control, feature-rich

#### F1 Model Features and Advantages
- **Forward Generation Method**: Generates intuitive movements in the normal generation direction (from first to last)
- **Simple Interface**: Omits Section (keyframe images) and Final (endframe) functions
- **Movement Richness**: Easier to generate dynamic movements compared to the standard version
- **Beginner-Friendly**: Fewer settings, intuitive operation
- **Image Influence**: Function to control change from the initial Image in the first section (adjustable from 100.0% to 102.0%)

#### Standard Model Features and Advantages
- **Reverse Generation Method**: Unique approach that generates from the final frame backwards
- **Feature-Rich Interface**: Complex UI allowing detailed settings
- **Keyframe Control**: Fine control using Image, Final, and section images
- **Advanced User-Oriented**: Enables sophisticated control through detailed settings

### Keyframe Settings *Added by nirvash
- **Image**: Main starting keyframe
- **Final Frame**: Final frame (optional)
- **Section Settings**: Individual keyframe images and prompts can be set for each section

### Automatic Keyframe Copy Function *Enhanced in v1.7
- **Red Frame (Section 0)**: Automatically copied to all even-numbered sections (0,2,4,6...)
- **Blue Frame (Section 1)**: Automatically copied to all odd-numbered sections (1,3,5,7...)
- **Toggle On/Off**: Can be toggled with a checkbox
- **Benefit**: Covers all sections with just two keyframe settings

### Section-Specific Prompts *Added in v1.2 [Experimental Implementation]
- Ability to set individual prompts for each section
- Section-specific prompts are only used during generation of that section
- Common prompt is used when left blank
- Note: This feature is an experimental implementation and its effectiveness is not guaranteed

### PNG Metadata Function *Added in v1.9.1
- Automatically embeds prompts, seed values, and section information in generated images
- Settings can be retrieved from saved images
- Standard metadata format compatible with SD-based tools
- Setting information can be shared along with images

### FP8 Optimization *Added in v1.9.1
- Reduces VRAM usage during LoRA application with 8-bit floating-point format
- Performance improvement with `scaled_mm` optimization for RTX 40 series GPUs
- Recommended to keep disabled by default (may result in warnings or errors in some environments)

### kisekaeichi Function *Added in v1.9.3 (oichi only)
- Video generation technology using reference images, conceived by furusu and implemented by Kohya
- Precise control using target index and history index
- Implementation of mask function for changing specific areas only
- Designed for use with FramePack-oichi (single frame inference)

### LoRA Preset Manager *Added in v1.9.3
- Save and manage up to 5 presets
- Easily switch presets from the UI
- Batch save LoRA files, strength, and other settings
- Contributes to workflow efficiency

### Hunyuan LoRA Settings *Added in v1.3, Significantly Improved in v1.9
- "Use LoRA" checkbox: Toggles LoRA on/off
- LoRA file selection: Select the LoRA file to use
- Application strength slider: Adjust LoRA influence from 0.0 to 1.0
- Format: Unified to HunyuanVideo format in v1.9, improving compatibility
- Note: When using LoRA, there may be a waiting time before the progress bar starts
- In v1.9, with code introduced from kohya-ss/FramePack-LoRAReady, both high VRAM and low VRAM modes use the same direct application method, significantly improving stability

### Output Folder Settings *Added in v1.2
- Specify output folder name
- "Save and Open Output Folder" button allows saving settings and opening the folder
- Settings are retained after restarting the application

### MP4 Compression Settings *Merged from Original in v1.6.2
- Adjustable from 0 to 100 range with slider (0=uncompressed, 16=default, higher values=higher compression/lower quality)
- Lower values result in higher quality but larger file sizes
- Setting to 16 may solve black screen issues

## 🧠 Understanding Keyframe Image Mechanisms and Concepts

### FramePack Operating Principle

FramePack's most distinctive feature is its unique "future to past" video generation approach. Conventional video generation AIs create frames sequentially from the first frame to the future, which can lead to quality degradation and reduced consistency in longer videos.

In FramePack, it first generates the final frame from the input image, then creates each frame in reverse direction. This allows it to maintain high quality and consistency even in longer videos.

**For the F1 model, generation happens in the conventional direction (from first to last).** This makes it easier to create dynamic movements, but the complexity of settings has been greatly simplified.

### FramePack-eichi's Enhanced Features

FramePack-eichi improves quality further by strategically placing multiple keyframe images:

1. **Preventing Abrupt Changes in Final Section**:
   - The original endframe with images set only for the first (last 1 second) section had an issue where the image changed suddenly in the final section (near the first 1 second)
   - FramePack-eichi takes a straightforward approach by allowing keyframe images to be set for all sections
   - Particularly important keyframes are highlighted with red frames, and setting images for these allows automatic copying
   - As mentioned above, FramePack generates videos from the final section, so both the section order and keyframe arrangement are set from the end
   - In 6-second mode, FramePack may not reach the keyframe images for loop mode before the loop ends
   - 8-second mode provides more gradual image transitions than 6-second mode
   - In any case (including for multiple scenes), the greater the difference between images, the larger the movement changes, resulting in smoother motion

2. **Loop Function Optimization**:
   - In loop mode, the first keyframe is automatically copied to the Final Frame
   - Since v1.5.1, copying from input images in normal mode has been stopped, and image copying is only enabled in loop mode
   - By setting the loop's starting pose in keyframe image 1, you can create smooth cyclic videos

3. **Section-Specific Prompt Settings**: *Added in v1.2 [Experimental Implementation]
   - Set unique prompts for each section to achieve different movements or expressions per section
   - For example, you can naturally express changes in movement like "walking" → "sitting" → "waving"
   - The influence of prompts is subtle, but effective when combined with keyframe images

4. **Short Video Modes**: *Added in v1.6.1
   - Support for 1-second, 2-second, 3-second, and 4-second short video modes
   - Each mode has optimized section numbers and copy patterns
   - Enables specialized control for short-duration expressions

### F1 Model Differences (Added in v1.9.1)

The F1 model has these major differences from the standard version:

1. **Forward Generation Approach**:
   - Generates from the first frame sequentially, allowing more natural movement transitions
   - Section (keyframe images) and Final (endframe) functions become unnecessary

2. **Simplified Interface**:
   - Only set the "Image" image
   - Adjust the degree of change in the first section with the "Image Influence" slider (100.0% to 102.0%)

3. **Operability and Results**:
   - High-quality results with fewer settings
   - Produces more movement and dynamic footage
   - Simple operation, user-friendly for beginners

### Basic Keyframe Image Relationships (Standard Version)

**Relationship between Image (input image), Final Frame (final frame), and keyframe images**:

1. **About Priority**:
   - Basically, except for the very last section, each section is based on the linearly generated previous section
   - If a keyframe image is set for the current section, it is used; if not, an intermediate state inferred from other images is used
   - **If a keyframe image is set for the final section, it takes priority over the Image.**

This structure allows for fine control over each section, resulting in more natural and consistent movements.

### v1.7 Innovation: Red/Blue Frame Keyframe System (Standard Version)

In v1.7, the keyframe image copy function was greatly improved, introducing a more efficient and intuitive system:

1. **Visual Distinction with Red/Blue Frames**:
   - **Red Frame (Section 0)**: Automatically copied to all even-numbered sections (0,2,4,6...)
   - **Blue Frame (Section 1)**: Automatically copied to all odd-numbered sections (1,3,5,7...)

2. **Keyframe Setting Efficiency**:
   - Cover all sections with just two keyframe settings
   - Previously required setting each section individually, now greatly streamlined with pattern-based automatic copying

3. **Dynamic Section Number Adaptation**:
   - Accurately calculates section numbers based on selected video length and frame size
   - Automatically adjusts copy destinations based on calculated section numbers

4. **Flexible Control with Checkbox**:
   - Easily toggle automatic keyframe copy function on/off
   - For complex videos, you can turn it off to control each section individually when needed

This system significantly reduces the effort of keyframe settings, especially when creating longer videos (10+ seconds).

### Prompt Setting Tips

Prompt settings are as important as keyframe images:

1. **Basic Prompt Structure**:
   - Writing in order of subject → movement → other elements is effective
   - Example: `The character walks gracefully, with clear movements, across the room.`

2. **Movement Specification Levels**:
   - No prompt: Almost no movement is generated
   - Simple movement: Basic movements are generated with prompts like `moves back and forth, side to side`
   - Specific movement: More complex movements are generated with detailed prompts like `dances powerfully, with clear movements, full of energy`

3. **Important Notes**:
   - Words representing large movements like "dance" may result in more exaggerated movements than expected
   - Practical prompt examples:
     - Gentle movement: `The character breathes calmly, with subtle body movements.`
     - Moderate movement: `The character walks forward, gestures with hands, with natural posture.`
     - Complex movement: `The character performs dynamic movements with energy and flowing motion.`

4. **Deep Structure of Prompts (LLAMA and CLIP Separation)**:
   - Inside FramePack, prompts are processed by two different models:
   
   - **LLAMA Model (256 token limit)**:
     - Responsible for detailed understanding of text and context processing
     - Used for controlling overall content and sequence of the video
     - Character count guideline: About 1000-1300 characters (English) or 200-400 characters (Japanese)
     - Relates to control of scene context and narrative
   
   - **CLIP Model (77 token limit)**:
     - Model specialized in associating images and text
     - Influences the generation of specific visual features in video frames
     - Character count guideline: About 300-400 characters (English) or 50-150 characters (Japanese)
     - Relates to control of style, subject, and visual attributes

5. **Effective Prompt Writing Strategy**:
   - **First 300-400 characters (English)/50-150 characters (Japanese)**:
     - Important "visual part" processed by both LLAMA and CLIP
     - Describe main visual elements, style, subject, and overall tone here
     - Example: `A young woman with long flowing hair, cinematic lighting, detailed facial features, soft expressions, gentle movements`
   
   - **Latter 600-900 characters (English)/150-250 characters (Japanese)**:
     - "Narrative part" processed only by LLAMA
     - Describe movement details, scene context, and sequence information here
     - Example: `The camera slowly pans from left to right. The woman gradually turns her head, her expressions changing from neutral to a slight smile. There is a sense of emotional buildup as if emotional music is playing in the background.`

6. **Section-Specific Prompt Usage**: *Added in v1.2 [Experimental Implementation]
   - Keep section-specific prompts concise, focusing on important movements for that section
   - Clear, specific instructions are more effective than long sentences
   - Example: Section 1 "walking motion", Section 2 "sitting motion", Section 3 "waving motion"
   - Note: The effect of section prompts is subtle, and combination with image settings is important

7. **F1 Model Prompts (v1.9.1)**:
   - Expressions that clearly specify movement are particularly effective
   - Specifying emotional expressions and movement speed concretely also yields good results
   - Example: `A character enthusiastically dancing with dynamic movements, arms swinging freely, head nodding to the rhythm, full of energy and life`

8. **Style Adjustment Using LoRA**: *Added in v1.3 [Experimental Implementation]
   - Combining LoRA selection and prompts can emphasize specific styles or expressions (probably)
   - LoRA effects can be adjusted with application strength (0.1-0.3 for subtle effects, 0.5-0.8 for pronounced effects)
   - Effects are maximized when prompts and LoRA selection match

### Selecting Effective Difference Images

The quality of FramePack's video generation is largely determined by keyframe image selection. Important points for choosing ideal difference images:

1. **Optimal Difference Level**:
   - **Excessively small differences**: Using almost identical images ("so-called intelligent differences") will generate almost no movement
   - **Excessively large differences**: Using completely unrelated images will not lead to natural movement
   - **Ideal differences**: Changes that AI can find relationships between, such as different poses of the same character, are optimal

2. **Maintaining Relevance**:
   - Simply flipping an image horizontally will be recognized by AI as a completely different image and won't lead to natural movement
   - Changes in face direction, hand position, and body pose are ideal difference elements
   - Maintaining consistency in background and clothing as much as possible allows AI to focus on character movement
   - Coincidentally, variations of a character created with similar prompts using image generation AI can be one of the ideal difference elements

3. **Characteristics of Ideal Difference Images**:
   - Same character with slightly changed posture
   - Subtle changes in expression (e.g., expressionless → smiling, however, without face position change, movement will be weak)
   - Pose changes with natural hand or arm movements
   - Gentle changes in head direction

4. **Experimental Approach**:
   - Selection of difference images has a stronger artistic aspect than scientific, so trial and error is important
   - Initially try differences with similar poses, then gradually adjust the magnitude of differences
   - Noting successful combinations allows application to future works

5. **Combination with Image Generation AI**:
   - If ideal difference images are not available, using image generation AI to create different poses of the same character is also effective
   - Keep pose changes specified in prompts modest and avoid dramatic changes for more natural movement

This structured approach helps maximize the strengths of both models and generate more expressive videos.

### F1 Image Influence vs Standard EndFrame Influence and All-Padding

#### F1 Model Image Influence (v1.9.1)

The F1 model introduces a new parameter called "Image Influence":

**What it affects**: This directly adjusts **the degree of change from the Image in the first section**.

**Technical mechanism**:
- Set in a very narrow range from 100.0% to 102.0%
- Setting to 100.0% maintains faithfulness to the Image
- As it approaches 102.0%, movement decreases, approaching almost static

**Effects**:
- At 100.0%, natural movement is generated
- Around 101.0%, changes from the initial frame become more gradual
- At 102.0%, almost no movement is generated

#### Standard Version EndFrame Influence and All-Padding

The standard FramePack-eichi has two important functions for controlling video movement: "EndFrame Influence Adjustment" and "All-Padding". While these may seem similar, their operating principles and effects are completely different.

##### 1. EndFrame Influence Adjustment (Introduced in v1.3)

**What it affects**: This directly changes **the strength of the final frame (Final Frame) itself**.

**Technical mechanism**:
- Exactly multiplies the specified value to the latent representation of the final frame
- Implemented in code as `modified_end_frame_latent = end_frame_latent * end_frame_strength`
- Value range is 0.01 to 1.00, with 1.00 being the default (no change)

**Effects**:
- Reducing the value from 1.0 to 0.5 makes the influence of the final frame "overall" exactly half
- Setting to 0.3 makes the influence of the final frame "overall" exactly 30%
- By weakening the direct influence of the final frame, characteristics of the first frame (Input Image) appear earlier

##### 2. All-Padding (Introduced in v1.4)

**What it affects**: This changes **how sections are connected to each other**.

**Technical mechanism**:
- Normally, padding values between sections are automatically calculated as `[3, 2, 2, 2, 1, 0]`
- When All-Padding is enabled, this value is unified to a single specified value (e.g., `[1.5, 1.5, 1.5, 1.5, 1.5, 0]`)
- Value range is 0.2 to 3.0, with 1.0 being the standard value
- The final section (0th) is always forced to 0

**Effects**:
- Setting to higher values like 1.5 makes each section reference the previous section more strongly, resulting in rarer changes
- Setting to lower values like 0.5 makes each section reference the previous section less, resulting in more changes
- The "distribution" of change amount is altered, but the overall strength of frames remains the same

#### Selection Guide for EndFrame Influence and All-Padding

Guidelines for appropriate settings based on the scene:

##### Appropriate Values for EndFrame Influence

- **Large difference images**: 0.3 to 0.6 (achieving gradual changes)
- **Moderate differences**: 0.5 to 0.8 (balanced transition)
- **Small differences**: 0.8 to 1.0 (near-standard influence)
- **Facial expression changes**: 0.7 to 0.8 (natural transition of expressions)
- **Large movements of body or hands**: 0.3 to 0.5 (more natural intermediate frames)

##### Appropriate Values for All-Padding

- **Smooth transitions**: 1.5 to 2.0 (pure transitions where section boundaries are not noticeable)
- **Standard movement**: 1.0 (balanced transition)
- **Active movement**: 0.5 to 0.7 (larger changes in each section)
- **Extreme movement**: 0.2 to 0.4 (very active and unpredictable movement)

##### Practical Application Techniques

- **Influence 0.5 + All-Padding 0.5**: For more dynamic movement
- **Influence 0.3 + Short-time mode**: When creating quick-change loop animations
- **Influence 0.8 + Long-time mode**: Expressing gentle movements with slow changes
- **Ultra-low influence (0.01 to 0.1)**: Almost ignoring the final frame and using a change of thinking where the first frame is the "goal"
- **High All-Padding value (2.0+) + Influence 0.5**: Weakens the influence of the final frame without making section boundaries noticeable

## 🛠️ Configuration Information

### Basic Settings (Windows bat Files)
- **Port setting**: `--port` parameter (Default: 8001)
  - Port number used by WebUI
  - Change if it conflicts with other applications
  
- **Server address**: `--server` parameter (Default: '127.0.0.1')
  - Change to `0.0.0.0` for access within local network
  
- **Auto browser launch**: `--inbrowser` option
  - Automatically opens browser at startup

### F1 Model Settings (Added in v1.9.1)
- **Image Influence**: Slider (Default: 100.0%)
  - Range: 100.0% to 102.0%
  - Adjusts the degree of change from the Image in the first section
  - Lower values: More active movement (100.0% is standard)
  - Higher values: More static movement (102.0% is almost stationary)

### Performance Settings
- **GPU memory preservation setting**: `gpu_memory_preservation` slider (Default: 10GB) *Existing feature
  - Lower value = More VRAM usage = Faster processing
  - Higher value = Less VRAM usage = Stable operation
  - Mechanism: The smaller the setting value, the more VRAM is released for the transformer model
  - Calculation method: Available VRAM for the tool is the upper limit (with margin) minus the setting value (minimum 6GB is always secured)
    - Example: With 16GB VRAM, assuming margin to 14GB, "14-(10-6)=10GB" is used
    - Setting to the lower limit of 6GB results in "14-(6-6)=14GB", using almost the upper limit (reduces processing time by about 10 seconds per section, but with memory swap risk)
  - Recommended values: 
    - 8GB VRAM: 7-8GB
    - 12GB VRAM: 6-8GB
    - 16GB or more: Around 6GB
  - Note: Increase the value if running other applications simultaneously
  - This tool secures a 3GB margin in the background for other image generation tools to run
  - When using LoRA, it's better to prepare an additional margin

- **High VRAM mode**: Auto-detection (v1.5.1: 60GB+ free VRAM, v1.6: 100GB+ free VRAM) *Feature improvement
  - When enabled: Models are always kept on GPU, reducing memory transfer overhead
  - Effect: Up to 20% processing speed improvement
  - In v1.6, the threshold was raised, making low VRAM mode used in most environments
  - Even in low VRAM mode, the same direct application method as high VRAM mode is used, improving feature consistency

### FP8 Optimization Settings (Added in v1.9.1)
- **FP8 optimization**: Checkbox (Default: Disabled)
  - When enabled: Applies quantization using 8-bit floating-point format
  - Effect: Significantly reduces VRAM usage, also improves processing speed on RTX 40 series GPUs
  - Note: May cause warnings or errors in some environments
  - Recommendation: Keep disabled normally, consider enabling only when using LoRA in low VRAM environments

### Generation Settings
- **Frame size setting**: `frame_size` dropdown menu (Default: 1 second) *Added in v1.5
  - 0.5 second: Generates frames for 0.5 seconds, section count and processing time almost double
  - By giving differences to images of each frame with All-padding 0 mode, even more intense movement is possible
  - 1 second: Generates frames for 1 second

- **Steps**: `steps` slider (Default: 25) *Existing feature
  - Increasing values improves quality but proportionally increases processing time
  - Recommended range: 20-30 (often similar quality can be achieved with 20)
  - 15 or less: Noticeable quality degradation occurs

- **TeaCache**: `use_teacache` checkbox (Default: Enabled) *Existing feature
  - Enabled: Processing approximately 15-20% faster
  - Side effect: May slightly degrade representation of hands and fingertips
  - Usage: Recommended to enable for general video generation, disable when fine detail is important

- **Random seed value**: `seed` numeric input or "Use Random Seed" checkbox *Feature added by nirvash
  - Same seed value = Reproducible results
  - Random seed: Generates different movements each time
  - Note: Results change with the same seed if prompts or images change

- **Distilled CFG Scale**: `gs` slider (Default: 10.0) *Existing feature
  - Distilled guidance scale value
  - Lower values = More free movement (increased deviation from prompt)
  - Higher values = More faithful to prompt (movement may be restricted)
  - Recommendation: Maintain default value (changing is for advanced users)

- **MP4 compression setting**: `mp4_crf` slider (Default: 16) *Merged from original in v1.6.2
  - Range: 0 to 100 (0=uncompressed, 100=maximum compression)
  - Lower values result in higher quality videos but larger file sizes
  - Higher values result in higher compression and smaller file sizes, but lower quality
  - Setting to 16 may solve black screen issues
  - Usage: Low values (0-10) for archival purposes, moderate values (16-30) for web sharing

### LoRA Settings (Added in v1.3, Significantly Improved in v1.9)
- **Use LoRA**: `use_lora` checkbox (Default: Disabled)
  - Enabled: Customizes the model using a LoRA file
  - When using LoRA, there may be longer waiting times before the counter starts

- **LoRA file**: File selection component
  - Specifies the LoRA file to use
  - Supported format: Unified to HunyuanVideo format in v1.9

- **LoRA strength**: `lora_strength` slider (Default: 0.8)
  - Range: 0.0 to 1.0
  - Lower values: Subtle effect
  - Higher values: Strong effect
  - Optimal value varies by LoRA file

- **Improvements in v1.9**
  - Significantly improved stability with code from kohya-ss/FramePack-LoRAReady
  - Unified application method for high VRAM and low VRAM modes
  - Reduced code complexity by using the common load_and_apply_lora function

### Frame Settings
- **Video length**: Radio buttons + `total_second_length` slider *Extension of unique feature
  - **eichi (Standard Version)**: 1-20 seconds, 30 seconds, 40 seconds (added in v1.9.3)
  - **F1 Version**: 1-20 seconds, 30 seconds, 40 seconds, 60 seconds, 120 seconds (added in v1.9.3)
  - Actual frame count is approximately 30fps equivalent

- **Automatic keyframe copy**: `enable_keyframe_copy` checkbox (Default: Disabled - Changed in v1.5.1) *Unique feature
  - Enabled: Keyframe images are automatically copied to other sections
  - Disabled: Each keyframe needs to be set individually
  - Usage: Advanced users designing complex movements may disable this

### Output Settings
- **Output folder**: Output folder setting field (Default: 'outputs') *Added in v1.2
  - Destination for generated videos and images
  - Folder name can be directly entered in the input field
  - "Save and Open Output Folder" button opens the folder
  - Settings are saved in JSON format and maintained after restart

- **Save section still images**: `save_section_frames` checkbox (Default: Disabled) *Feature added by nirvash
  - Enabled: Final frame of each section is saved as a still image
  - Usage: Useful for checking connection points between sections

- **Save section videos**: `keep_section_videos` checkbox (Default: Disabled) *Unique feature
  - Enabled: Video files for each section are retained, will remain if ended with "End"
  - Disabled: Only the final completed video is saved (intermediate files are deleted), note they don't go to trash
  - Usage: Useful for checking movements in each section individually

- **Save tensor data**: Checkbox (Default: Disabled) *Added in v1.8
  - Enabled: Saves latent representations of generated videos in .safetensors format
  - Usage: Useful for later combining with other videos or saving the generation process

### Prompt Management
- **Preset save**: "Save" button
  - Saves current prompt with a name
  - Setting a blank name sets it as the default prompt at startup
  
- **Apply preset**: "Apply" button
  - Applies the selected preset's prompt to current generation settings
  
- **Preset management**:
  - Delete: Removes unwanted presets (default preset cannot be deleted)
  - Clear: Clears the editing field

### PNG Metadata Settings (Added in v1.9.1)
- **Metadata embedding**: Automatically enabled
  - Automatically saves prompts, seed values, and section information in generated images
  - Standard metadata format compatible with SD-based tools
  - Setting information can be shared along with images

### Docker Settings (Added in v1.9.1)
- **Language selection**: Set with command parameter in docker-compose.yml
  ```yaml
  # For Japanese:
  command: ["--lang", "ja"]
  
  # For Traditional Chinese:
  command: ["--lang", "zh-tw"]
  
  # For English (default):
  command: ["--lang", "en"]
  
  # For Russian (added in v1.9.3):
  command: ["--lang", "ru"]
  ```

- **Volume settings**: Set with volumes parameter in docker-compose.yml
  ```yaml
  volumes:
    - ./data:/app/framepack/data
    - ./models:/app/framepack/hf_download
    - ./outputs:/app/framepack/outputs
  ```
  - data: Storage location for configuration files
  - models: Location for downloaded models
  - outputs: Storage location for generated videos

### Multi-language Support (Extended in v1.9.3)
FramePack-eichi supports the following 4 languages:

- **Japanese**: `--lang ja` or launch script `run_endframe_ichi.bat`
- **English**: `--lang en` or launch script `run_endframe_ichi_en.bat`
- **Traditional Chinese**: `--lang zh-tw` or launch script `run_endframe_ichi_zh-tw.bat`
- **Russian**: `--lang ru` or launch script `run_endframe_ichi_ru.bat` *Added in v1.9.3

By using the language-specific launch scripts, the UI will automatically display in the appropriate language at startup. The language setting ensures all UI labels, descriptions, and help texts are displayed in the selected language.