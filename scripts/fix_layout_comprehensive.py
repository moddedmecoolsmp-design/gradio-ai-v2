#!/usr/bin/env python3
"""
Comprehensive fix for app.py UI layout
"""

def fix_layout():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Key line numbers (based on previous analysis)
    # Line 3261: with gr.Row()  (indent=12)
    # Line 3262: with gr.Column(scale=1)  (indent=16) - LEFT COLUMN START
    # Line 3379: LEFT COLUMN CLOSES (implicitly)
    # Line 3380-3656: Components at indent=12 (WRONG - should be indent=16+ in left column)
    # Line 3658-3663: Generate button + seed_info (currently indent=20 - WRONG)
    # Line 3664-3697: Right column (currently indent=16 - WRONG, should be same as left column)
    # Line 3698: Audio Tools tab (indent=8)

    # Strategy: Re-indent everything between 3262 and 3697 to be inside the Row at 3261
    # Left column: 3262-3663 (need to add 4 spaces to lines 3380-3663)
    # Right column: 3664-3697 (already correct indent but wrong structure)

    new_lines = []
    in_left_column = False
    left_column_start = 3262
    left_column_content_wrong_indent_start = 3380
    generate_btn_section_start = 3658
    right_column_start = 3664
    audio_tools_tab = 3698

    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed

        # Before left column starts - keep as is
        if line_num < left_column_start:
            new_lines.append(line)

        # Left column content that's correctly indented (3262-3379)
        elif line_num >= left_column_start and line_num < left_column_content_wrong_indent_start:
            new_lines.append(line)

        # Left column content that's WRONGLY indented at 12 (should be 16+)
        # Lines 3380-3657 need +4 spaces
        elif line_num >= left_column_content_wrong_indent_start and line_num < generate_btn_section_start:
            # Add 4 spaces to indent these lines inside the left column
            if line.strip():  # Only add indent to non-empty lines
                indent = len(line) - len(line.lstrip())
                new_lines.append('    ' + line)  # Add 4 spaces
            else:
                new_lines.append(line)  # Keep blank lines as-is

        # Generate button section (3658-3663) - currently at wrong indent
        # Should be inside left column at indent=16 for "with gr.Row()"
        elif line_num >= generate_btn_section_start and line_num < right_column_start:
            # These lines have too much indent, remove 4 spaces
            if line.startswith('    '):
                new_lines.append(line[4:])  # Remove 4 spaces
            else:
                new_lines.append(line)

        # Right column (3664-3697) - structure is wrong
        # Line 3664 should close left column and start right column
        elif line_num == right_column_start:
            # Don't include the malformed "with gr.Column(scale=1):" line
            # Instead, output proper right column start
            new_lines.append('                with gr.Column(scale=1):\n')

        # Right column content (3665-3697)
        elif line_num > right_column_start and line_num < audio_tools_tab:
            new_lines.append(line)  # Keep as-is (already correct indent)

        # Audio Tools tab and beyond - keep as-is
        else:
            new_lines.append(line)

    # Write fixed file
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Fixed! Original: {len(lines)} lines, New: {len(new_lines)} lines")

if __name__ == '__main__':
    fix_layout()
