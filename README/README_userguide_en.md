# FramePack-eichi User Guide | [日本語](README_userguide.md) | [繁體中文](README_userguide_zh.md) | [Русский](README_userguide_ru.md)

This guide provides detailed instructions on how to use both the Standard and F1 versions of FramePack-eichi. Each has different features and uses, so choose the optimal model based on your purpose. For basic information and installation methods, please refer to the [README](README_en.md); for detailed settings, check the [Extended Documentation](README_column_en.md); and for update history, see the [Changelog](README_changelog.md).

## Table of Contents

- [What is FramePack-eichi](#what-is-framepack-eichi)
- [Differences Between Standard and F1 Versions](#differences-between-standard-and-f1-versions)
- [How to Use the Standard Version](#how-to-use-the-standard-version)
- [How to Use the F1 Version](#how-to-use-the-f1-version)
- [How to Use the oichi Version (One-Frame Version)](#how-to-use-the-oichi-version-one-frame-version)
- [Detailed Explanation of Settings](#detailed-explanation-of-settings)
- [Troubleshooting](#troubleshooting)

## What is FramePack-eichi

FramePack-eichi is an AI tool that generates videos from images. It enables the expression of "movement" that is not possible with conventional image generation AI. You can generate short videos from still images or create animations based on multiple keyframes.

This tool has two versions:

1. **Standard Version**: Model specialized in keyframe control
2. **F1 Version**: Model specialized in forward generation

## Differences Between Standard and F1 Versions

|Feature|Standard Version|F1 Version|
|---|---|---|
|**Basic Concept**|Precise movement specification through keyframe control|Natural and active movement through forward generation|
|**Keyframes**|Multiple section images can be specified|Uses only the first image|
|**Final Frame**|Can be explicitly specified|Automatic generation only|
|**Image Influence**|1.0 to 100.0%|100.0 to 102.0%|
|**Suitable Uses**|When you want to control exact movements|When you want to generate natural, unpredictable movements|
|**Customizability**|High (detailed control of each section)|Low (specify overall atmosphere)|
|**Learning Curve**|Somewhat high (need to understand multiple parameters)|Low (simple operability)|

## How to Use the Standard Version

### 1. How to Launch

To use the Standard version, run one of the following batch files:

- Japanese version: `run_endframe_ichi.bat`
- English version: `run_endframe_ichi_en.bat`
- Chinese version: `run_endframe_ichi_zh-tw.bat`
- Russian version: `run_endframe_ichi_ru.bat`

### 2. Basic Usage

1. **Selecting Video Mode**:
   - Select the desired duration mode from the "Video Mode" dropdown in the upper right
   - Available modes for Standard version: 1-20 seconds, 30 seconds, 40 seconds *Extended in v1.9.3
   - The mode determines the number of sections (e.g., "4 seconds" is 4 sections)

2. **Setting Keyframe Images**:
   - Select tabs "0" to "number of sections-1" to set images for each section
   - Click the "Image" button to upload an image (or use "Clipboard" to paste from clipboard)
   - Check the red/blue frames to understand the role of the image as a keyframe (indicating copying to even/odd sections)

3. **Final Frame (Optional)**:
   - Select the "Final" tab
   - Click the "Image" button to upload a final frame image (or from clipboard)

4. **Setting Prompts**:
   - You can set prompts for each section (text box under the section tabs)
   - Set the base prompt common to all sections in the "Base" tab
   - Use "Negative Prompt" to exclude unwanted elements

5. **Detailed Settings**:
   - "EndFrame Influence": Adjust the influence of the final frame (0.01 to 1.00)
   - "Keyframe Copy": Turn automatic copy function on/off
   - "Seed": Set the random seed value (-1 for random)

6. **LoRA Settings (Optional)**:
   - Apply a LoRA model to adjust expressions
   - Enter the path to a .safetensors file under "path1:"
   - Set the weight to a value between 0.0 and 1.0 in "weight1:"
   - **Preset Manager *New in v1.9.3**:
     - Save and manage up to 5 presets
     - Easily switch presets from the UI
     - Batch save LoRA files, strength, and other settings

7. **Video Generation**:
   - Click the "Start" button after completing all settings
   - Once the generation process is complete, the video is saved to the output folder and played automatically

### 3. Advanced Usage

#### Bulk Section Information Addition

- You can set multiple section information at once using a zip file
- Zip file structure:
  ```
  zipname.zip
  ├── anydir/ (only one folder level or none is allowed)
  │ ├── sections.yaml (required)
  │ ├── start.png (starting frame, optional. Filename must start with "start*")
  │ ├── end.png (final frame, optional. Filename must start with "end*")
  │ ├── 000.png (section 0 image. Filename must start with a 3-digit number)
  │ ├── 001.png (section 1 image. Filename must start with a 3-digit number)
  ```
- sections.yaml format:
  ```yaml
  section_info:
    - section: 0
      prompt: "Character reaches out"
    - section: 1
      prompt: "Character turns toward us"
  ```
- When uploaded, section images and prompts are automatically set

#### Utilizing the Keyframe Copy Function

- When "Keyframe Copy" is on, section 0 (red frame) images are copied to even-numbered sections (0, 2, 4...), and section 1 (blue frame) images are copied to odd-numbered sections (1, 3, 5...)
- This allows you to efficiently generate videos with minimal image settings
- Especially effective for long video modes (10+ seconds)

#### Section-Specific Prompt Settings

- The prompt for each section determines the characteristics of that section
- Since it's used in combination with the base prompt, it's sufficient to describe only section-specific features
- Including words that express changes in movement (e.g., "run" → "jump" → "sit") is effective

#### Adjusting EndFrame Influence

- High values (close to 1.00): Strong convergence to the final frame, tends to result in linear movement
- Low values (close to 0.01): Weak convergence to the final frame, tends to result in more free movement
- For natural movement, values around 0.1 to 0.3 are recommended

## How to Use the F1 Version

### 1. How to Launch

To use the F1 version, run one of the following batch files:

- Japanese version: `run_endframe_ichi_f1.bat`
- English version: `run_endframe_ichi_en_f1.bat`
- Chinese version: `run_endframe_ichi_zh-tw_f1.bat`
- Russian version: `run_endframe_ichi_f1_ru.bat`

### 2. Basic Usage

1. **Selecting Video Mode**:
   - Select the desired duration mode from the "Video Mode" dropdown in the upper right
   - Available modes for F1 version: 1-20 seconds, 30 seconds, 40 seconds, 60 seconds, 120 seconds *Extended in v1.9.3
   - In the F1 version, only the first image is used as a section image

2. **Setting the Initial Image**:
   - Click the "Image" button to upload an image (or use "Clipboard" to paste from clipboard)
   - This image becomes the first frame of the video, and subsequent frames are automatically generated

3. **Setting Prompts**:
   - Set the prompt in the "Base" tab (only this prompt is used in the F1 version)
   - Use "Negative Prompt" to exclude unwanted elements

4. **Detailed Settings**:
   - "Image Influence": Adjust the influence of the initial image (100.0 to 102.0%)
   - "Seed": Set the random seed value (-1 for random)

5. **LoRA Settings (Optional)**:
   - Apply a LoRA model to adjust expressions
   - Enter the path to a .safetensors file under "path1:"
   - Set the weight to a value between 0.0 and 1.0 in "weight1:"
   - **Preset Manager *New in v1.9.3**:
     - Save and manage up to 5 presets
     - Easily switch presets from the UI
     - Batch save LoRA files, strength, and other settings

6. **Video Generation**:
   - Click the "Start" button after completing all settings
   - Once the generation process is complete, the video is saved to the output folder and played automatically

### 3. F1 Version Tips

#### Appropriate Adjustment of Image Influence

- **100.0%**: Generates videos that are very faithful to the initial image. Characters and composition don't change much, providing the most stability.
- **100.5%**: A balanced setting with some change and stability.
- **101.0 to 102.0%**: Results in bolder movements and changes, but may deviate from the initial image.

#### Guiding Movement with Prompts

- Use verbs and action words to guide specific movements
  - Examples: "walking", "running", "jumping", "dancing", etc.
- You can specify not only character movements but also camera work
  - Examples: "zoom in", "pan left", "closeup", etc.

#### Understanding F1 Version Characteristics

- The new F1 version is "FramePack_F1_I2V_HY_20250503", a model specialized in forward generation
- Unlike the Standard version, you can't specify fine movements, but more natural and unpredictable movements are possible
- It's effective to use it with the mindset of specifying a general direction with prompts and leaving the details to the AI

## How to Use the oichi Version (One-Frame Version)

### 1. Overview

The oichi version is a special version that generates the next frame from a single image. It specializes in predicting one frame ahead rather than generating entire videos, allowing for lighter and faster processing. In v1.9.3, the kisekaeichi function was added.

### 2. How to Launch

To use the oichi version, run one of the following batch files:

- Japanese version: `run_oneframe_ichi.bat`
- English version: `run_oneframe_ichi_en.bat`
- Chinese version: `run_oneframe_ichi_zh-tw.bat`
- Russian version: `run_oneframe_ichi_ru.bat`

### 3. Basic Usage

1. **Image Setup**:
   - Upload a single image
   - The next frame will be generated from this image

2. **Prompt Setup**:
   - Describe the content of the next frame you want to generate
   - Specify movements or changes in detail

3. **kisekaeichi Function *New in v1.9.3**:
   - Technology using reference images conceived by furusu and implemented by Kohya
   - Precise control using target index and history index
   - Mask function for changing only specific areas
   - Used when applying different costumes or styles

4. **Running Generation**:
   - Click the "Generate" button
   - The next frame will be generated

### 4. kisekaeichi Function Details

- **Reference Image Setup**: Set a reference image for the style or costume you want to change
- **Mask Function**: Specify the area you want to change (overall or partial changes possible)
- **Strength Adjustment**: Adjust the influence of the reference image
- **Note**: As an experimental feature, results heavily depend on the input and reference images

## Detailed Explanation of Settings

### Common Settings

#### Prompt-Related
- **Prompt**: Text that specifies the content of the video to be generated. Describe in detail the elements and features you want to use.
- **Negative Prompt**: Text that specifies elements to avoid. Including terms like "low quality", "blurry", etc. improves quality.

#### Technical Settings
- **Seed**: Random seed value. Using the same value yields similar results. Use "-1" for random.
- **FP8 Optimization**: Option to reduce VRAM usage during LoRA application. Especially effective on RTX 40 series GPUs.
- **Scale Processing**: Specify how to adjust video resolution.
- **Output Folder**: Destination for generated files.
- **MP4 Quality**: Set video compression rate (0-100, lower is higher quality, higher is lighter weight).

#### LoRA Settings
- **LoRA Path**: File path to the LoRA model to use.
- **LoRA Weight**: Influence of each LoRA model (0.0 to 1.0).

### Standard Version Specific Settings

- **Keyframe Copy**: When on, automatically copies images in specific patterns.
- **EndFrame Influence**: Adjust the influence of the final frame (0.01 to 1.00).
- **Section Tabs (0, 1, 2...)**: Settings screens for each keyframe.
- **Final Tab**: Settings screen for the final frame.

### F1 Version Specific Settings

- **Image Influence**: Influence of the initial image (100.0 to 102.0%)
  - 100.0%: Very faithful to the initial image
  - 100.5%: Balanced change
  - 101.0 to 102.0%: Bolder changes and movements

## Troubleshooting

### Common Problems

#### Black Screen is Generated
- **Cause**: GPU VRAM shortage, batch size or process conflicts
- **Solutions**:
  - Terminate other GPU processes
  - Turn on FP8 optimization
  - Set MP4 quality value to around 16
  - Use smaller video modes (1 second, 2 seconds, etc.)

#### Error: CUDA out of memory
- **Cause**: GPU VRAM shortage
- **Solutions**:
  - Turn on FP8 optimization
  - Use smaller video modes
  - Terminate other GPU processes
  - Use images with lower resolution

#### Prompts Not Properly Reflected
- **Cause**: Unclear prompt priority or expression
- **Solutions**:
  - Place important elements at the beginning
  - Use parentheses or emphasis syntax (e.g., (important element))
  - Use more specific words
  - For the Standard version, check the combination of section-specific prompts and base prompt

### Standard Version Specific Problems

#### Unnatural Changes Between Sections
- **Cause**: Gap between keyframes is too large
- **Solutions**:
  - Use keyframe images with more similar features
  - Divide changes gradually across multiple sections
  - Adjust EndFrame influence

#### Keyframes Being Ignored
- **Cause**: Image influence is too low
- **Solutions**:
  - Increase Image influence (50% or more recommended)
  - Emphasize keyframe features in the prompt

### F1 Version Specific Problems

#### Large Deviation from Initial Image
- **Cause**: Image influence is too high
- **Solutions**:
  - Lower Image influence (100.0 to 100.5% recommended)
  - Describe more specifically in the prompt

#### Little/Monotonous Movement
- **Cause**: Lack of expressions indicating movement in the prompt
- **Solutions**:
  - Add verbs and action words
  - Increase Image influence (101.0 to 102.0%)
  - Select a longer video mode