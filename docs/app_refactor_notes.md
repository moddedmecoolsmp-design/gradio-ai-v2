# App.py Refactor Analysis

## Current Status
- Audio Tools tab already exists at top level (line 3823)
- Models only download when generate_image/batch/video functions are called
- UI works but has some indentation inconsistencies (non-breaking)

## User's Actual Issue
User reported: "why is flux klein running if im not using a picture model"

This means models were downloaded even though they only wanted audio features.

## Root Cause
The user likely either:
1. Opened the "Image Generation" tab which might have triggered something
2. Or there's a default behavior we're missing

## Solution
1. Add clear warnings about model downloads
2. Ensure Audio Tools tab is completely isolated
3. Fix minor indentation issues for code cleanliness
4. Add "lazy loading" indicators

## Implementation Plan
1. Add banner to Image Generation tab: "Models will download (~3-24GB) when you first click Generate"
2. Ensure no model-related code runs when Audio Tools tab is active
3. Clean up indentation in UI section (lines 3385-3900)
4. Test that Audio Tools work without any image model downloads
