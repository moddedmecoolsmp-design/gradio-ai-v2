"""
Gender detection and preservation for FLUX models.
Uses InsightFace for detection, prompt engineering for preservation.
"""
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
    return iou

def deduplicate_faces(faces, iou_threshold=0.5):
    """Remove overlapping face detections to prevent duplicates."""
    if not faces:
        return []

    # Sort faces by detection score descending
    sorted_faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
    keep = []

    for i, face in enumerate(sorted_faces):
        is_duplicate = False
        for kept_face in keep:
            if calculate_iou(face.bbox, kept_face.bbox) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(face)

    return keep

def detect_gender_from_image(
    image: Image.Image,
    face_app
) -> Tuple[str, float, int]:
    """
    Detect gender(s) from image using InsightFace.
    """
    img_np = np.array(image.convert('RGB'))
    faces = face_app.get(img_np)
    faces = deduplicate_faces(faces)

    if not faces:
        return "unknown", 0.0, 0

    male_count = sum(1 for f in faces if f.gender == 1)
    female_count = len(faces) - male_count

    if male_count > female_count:
        dominant = "male"
    elif female_count > male_count:
        dominant = "female"
    else:
        dominant = "mixed"

    avg_conf = sum(f.det_score for f in faces) / len(faces)
    return dominant, float(avg_conf), len(faces)

def get_gender_details(
    image: Image.Image,
    face_app,
    visualize: bool = False
) -> dict:
    """
    Get detailed gender breakdown and optionally a visualized detection image.
    """
    img_np = np.array(image.convert('RGB'))
    faces = face_app.get(img_np)
    faces = deduplicate_faces(faces)

    if not faces:
        return {
            'male_count': 0,
            'female_count': 0,
            'total_faces': 0,
            'dominant_gender': 'unknown',
            'gender_description': '',
            'visualization': None
        }

    male_count = sum(1 for f in faces if f.gender == 1)
    female_count = len(faces) - male_count

    # Build prompt description
    if male_count > 0 and female_count > 0:
        parts = []
        if male_count == 1: parts.append("1 man")
        else: parts.append(f"{male_count} men")
        if female_count == 1: parts.append("1 woman")
        else: parts.append(f"{female_count} women")
        desc = f"({' and '.join(parts)}:1.2)"
    elif male_count > 0:
        desc = f"({'man' if male_count == 1 else f'{male_count} men'}:1.2), (male:1.1)"
    else:
        desc = f"({'woman' if female_count == 1 else f'{female_count} women'}:1.2), (female:1.1)"

    dominant = "male" if male_count > female_count else "female" if female_count > male_count else "mixed"

    visualization = None
    if visualize:
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)

        for i, face in enumerate(faces):
            box = face.bbox.astype(int)
            gender = "Male" if face.gender == 1 else "Female"
            color = (0, 122, 255) if face.gender == 1 else (255, 45, 85) # Blue for male, Red/Pink for female

            # Draw box
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)

            # Draw label background
            label = f"{gender} ({face.det_score:.2f})"
            label_pos = [box[0], box[1] - 20]
            draw.rectangle([label_pos[0], label_pos[1], label_pos[0] + 120, label_pos[1] + 20], fill=color)
            draw.text((label_pos[0] + 5, label_pos[1] + 2), label, fill=(255, 255, 255))

        visualization = vis_img

    return {
        'male_count': male_count,
        'female_count': female_count,
        'total_faces': len(faces),
        'dominant_gender': dominant,
        'gender_description': desc,
        'visualization': visualization
    }

def enhance_prompt_with_gender(prompt: str, gender_info: dict, strength: float = 1.0) -> str:
    if gender_info['total_faces'] == 0:
        return prompt
    desc = gender_info['gender_description']
    if strength != 1.0:
        def adjust(match):
            val = float(match.group(1)) * strength
            return f":{val:.1f})"
        desc = re.sub(r':(\d+\.?\d*)\)', adjust, desc)
    return f"{desc}, {prompt}"

def get_gender_negative_prompt(gender_info: dict, strength: float = 1.3) -> str:
    if gender_info['total_faces'] == 0:
        return ""
    male_count = gender_info['male_count']
    female_count = gender_info['female_count']
    negatives = []
    if male_count > 0 and female_count == 0:
        negatives.extend([f"(female:{strength})", f"(woman:{strength})", "(breasts:1.5)"])
    elif female_count > 0 and male_count == 0:
        negatives.extend([f"(male:{strength})", f"(man:{strength})", "(beard:1.5)"])
    return ", ".join(negatives)

def merge_negative_prompts(original: str, gender_neg: str) -> str:
    if not original: return gender_neg
    if not gender_neg: return original
    return f"{original}, {gender_neg}"

def get_cached_face_app(device: str = "cuda"):
    from src.image.face_analysis_provider import get_face_analysis
    return get_face_analysis(device=device, name='buffalo_l')

def unload_face_app():
    from src.image.face_analysis_provider import unload_face_analysis
    unload_face_analysis()
