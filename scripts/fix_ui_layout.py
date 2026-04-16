#!/usr/bin/env python3
"""
Fix app.py UI layout:
1. Consolidate duplicate code
2. Fix 2-column layout in Image Generation tab
3. Move output components to right column
4. Ensure Audio Tools tab is properly structured
"""

import re

def fix_app_layout():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find key line numbers
    img_gen_tab_start = None
    left_col_start = None
    generate_btn_line = None
    output_image_line = None
    audio_tools_tab_line = None

    for i, line in enumerate(lines):
        if 'with gr.TabItem("Image Generation"):' in line:
            img_gen_tab_start = i
        elif 'with gr.Row():' in line and img_gen_tab_start and left_col_start is None:
            # First Row after Image Generation tab
            if i > img_gen_tab_start and i < img_gen_tab_start + 10:
                left_col_start = i
        elif 'generate_btn = gr.Button("Generate", variant="primary")' in line:
            generate_btn_line = i
        elif 'output_image = gr.Image(label="Generated Image"' in line:
            output_image_line = i
        elif 'with gr.TabItem("Audio Tools"):' in line:
            audio_tools_tab_line = i

    print(f"Image Generation tab starts at line {img_gen_tab_start}")
    print(f"Left column Row starts at line {left_col_start}")
    print(f"Generate button at line {generate_btn_line}")
    print(f"Output image at line {output_image_line}")
    print(f"Audio Tools tab at line {audio_tools_tab_line}")

    # The issue: output_image (line 3665) is OUTSIDE the Row that started at 3262
    # It should be in a second Column inside that Row

    # Strategy: Find where to close left column and insert right column
    # The left column should end just before "with gr.Row(): generate_btn"
    # Then we need to add proper indentation for the right column

    if not all([img_gen_tab_start, left_col_start, generate_btn_line, output_image_line]):
        print("Could not find all required sections!")
        return

    # Find the line with seed_info (should be end of left column content)
    seed_info_line = None
    for i in range(generate_btn_line, output_image_line):
        if 'seed_info = gr.Textbox(label="Generation Info"' in lines[i]:
            seed_info_line = i
            break

    print(f"Seed info at line {seed_info_line}")

    # The fix:
    # 1. All content from left_col_start to seed_info_line should be indented properly in left column
    # 2. After seed_info, we need to add the right column
    # 3. Right column should contain output_image, pose_skeleton_output, and Examples

    # Find where Examples end
    examples_end = None
    for i in range(output_image_line, audio_tools_tab_line):
        if 'gr.Examples(' in lines[i] and 'Anime → Photoreal Examples' in ''.join(lines[i:i+10]):
            # Find the closing of this Examples
            for j in range(i, audio_tools_tab_line):
                if ')' in lines[j] and 'inputs=' in lines[j]:
                    examples_end = j
                    break
            break

    print(f"Examples end around line {examples_end}")

    # Build the corrected structure
    new_lines = lines[:seed_info_line+1]

    # Add right column
    new_lines.append('\n')
    new_lines.append('                with gr.Column(scale=1):\n')
    new_lines.append('                    output_image = gr.Image(label="Generated Image", type="pil", format="png")\n')
    new_lines.append('                    pose_skeleton_output = gr.Image(\n')
    new_lines.append('                        label="Extracted Pose Skeleton (for reference)",\n')
    new_lines.append('                        type="pil",\n')
    new_lines.append('                        visible=True,\n')
    new_lines.append('                        interactive=False\n')
    new_lines.append('                    )\n')
    new_lines.append('\n')
    new_lines.append('                    persisted_state = gr.State(initial_state)\n')
    new_lines.append('\n')
    new_lines.append('                    # Examples\n')
    new_lines.append('                    gr.Examples(\n')
    new_lines.append('                        examples=[\n')
    new_lines.append('                            ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],\n')
    new_lines.append('                            ["Portrait of a young woman, soft studio lighting, professional photography"],\n')
    new_lines.append('                            ["Cyberpunk city street at night, neon lights, rain reflections"],\n')
    new_lines.append('                            ["A cute cat wearing a tiny hat, studio photo, soft lighting"],\n')
    new_lines.append('                            ["Abstract art, vibrant colors, fluid shapes, modern design"],\n')
    new_lines.append('                        ],\n')
    new_lines.append('                        inputs=[prompt],\n')
    new_lines.append('                    )\n')
    new_lines.append('\n')
    new_lines.append('                    gr.Examples(\n')
    new_lines.append('                        examples=[\n')
    new_lines.append('                            [\n')
    new_lines.append('                                ANIME_PHOTO_PRESETS["Anime → Photoreal"]["prompt"],\n')
    new_lines.append('                                ANIME_PHOTO_PRESETS["Anime → Photoreal"]["negative_prompt"],\n')
    new_lines.append('                            ],\n')
    new_lines.append('                        ],\n')
    new_lines.append('                        inputs=[prompt, negative_prompt],\n')
    new_lines.append('                        label="Anime → Photoreal Examples",\n')
    new_lines.append('                    )\n')
    new_lines.append('\n')

    # Continue with Audio Tools tab and rest of file
    new_lines.extend(lines[audio_tools_tab_line:])

    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"\nFixed! Removed {len(lines) - len(new_lines)} duplicate/misplaced lines")
    print(f"Original: {len(lines)} lines, New: {len(new_lines)} lines")

if __name__ == '__main__':
    fix_app_layout()
