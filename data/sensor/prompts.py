"""
Activity-identification prompts for wearable-sensor input, shared by the lmms_eval
task and the fine-tuning export. Same labels and FINAL_ANSWER format as the video
prompts in ``lmms_eval/tasks/strokerehab/utils_identification.py``; the class
descriptions are rewritten in terms of what the sensors measure (median trial
durations and joint-angle patterns taken from the training split).
"""

import json

from data.sensor.render import TEXT_LEGEND

ACTIVITY_NAMES = {
    "brushing": "Brushing",
    "combing": "Combing",
    "deodorant": "Deodorant",
    "drinking": "Drinking",
    "face wash": "Face wash",
    "feeding": "Feeding",
    "glasses": "Glasses",
    "rtt exercise": "RTT exercise",
    "shelf exercise": "Shelf exercise",
}

_PLAIN_CLASSES = {
    "Brushing": "The patient applies toothpaste to a toothbrush, brushes their teeth, rinses, and sets the brush back down.",
    "Combing": "The patient picks up a comb and combs both sides of their hair.",
    "Deodorant": "The patient twists open a deodorant stick, applies it under the arm, then replaces the cap.",
    "Drinking": "The patient pours water from a bottle into a cup, takes a sip, and replaces the cap.",
    "Face wash": "The patient washes and dries their face using two washcloths at a sink.",
    "Feeding": "The patient prepares bread with margarine on a plate and eats a small piece using utensils.",
    "Glasses": "The patient puts on or removes a pair of glasses from the tabletop.",
    "RTT exercise": "The patient slides a toilet paper roll between center and outer targets on a flat surface.",
    "Shelf exercise": "The patient transfers a toilet paper roll between the center target and multiple shelf levels.",
}

_KINEMATIC_CLASSES = {
    "Brushing": "Long (about 70 s). Both elbows are often flexed past 100 degrees (hand at the mouth) with small, fast hand oscillations while brushing.",
    "Combing": "Very short (about 15 s). The paretic arm is raised to the head: high shoulder flexion and elbow flexion near 100 degrees, with repeated strokes.",
    "Deodorant": "Short (about 25 s). Bimanual; elbows stay below about 90 degrees while one arm lifts away from the body (shoulder abduction) and the hands twist the cap.",
    "Drinking": "Short (about 30 s). Pouring rotates the wrist; the cup is brought to the mouth, briefly flexing one elbow to about 95 degrees.",
    "Face wash": "Medium (about 45 s). Both hands repeatedly go to the face (both elbows above 100 degrees) while leaning over a sink.",
    "Feeding": "Longest (about 90 s). Bimanual fine movements at a table (spreading, cutting) with the trunk leaning forward (about 15 degrees thoracic flexion); elbows mostly below 90 degrees.",
    "Glasses": "Short (about 20 s). Both hands are raised to the face once or twice (elbows above 100 degrees).",
    "RTT exercise": "About 65 s. Only the paretic arm works while the non-paretic arm stays still; repeated slow slides along the table with low hand acceleration and moderate shoulder flexion.",
    "Shelf exercise": "About 55 s. Only the paretic arm works while the non-paretic arm stays still; repeated lifts to shelves with large shoulder flexion (about 80 degrees) and abduction peaks.",
}

_INPUT_DESCRIPTIONS = {
    "image": (
        "The image plots wearable inertial-sensor recordings from a stroke patient performing one activity. "
        "Panels from top to bottom: paretic (affected) arm joint angles, non-paretic arm joint angles, "
        "trunk flexion angles, and hand acceleration magnitude (gravity removed). The x-axis is time in seconds."
    ),
    "text": (
        "The table below lists wearable inertial-sensor recordings from a stroke patient performing one activity. "
        + TEXT_LEGEND
    ),
}


def build_prompt(mode: str = "image", style: str = "kinematic", sensor_text: str | None = None) -> str:
    """mode: "image" or "text"; style: "plain" (video descriptions) or "kinematic" (sensor hints)."""
    classes = {
        name: f"{_PLAIN_CLASSES[name]} {_KINEMATIC_CLASSES[name]}" if style == "kinematic" else _PLAIN_CLASSES[name]
        for name in _PLAIN_CLASSES
    }
    prompt = json.dumps({
        "input": _INPUT_DESCRIPTIONS[mode],
        "question": "Which activity is the patient performing in this recording?",
        "response_instructions": "After noting your observations, end your reply with exactly one line: FINAL_ANSWER: <activity_name>",
        "instructions": "Determine the main activity being performed by the patient. Choose the most fitting activity label from the list below. Always respond with an activity, even if uncertain.",
        "activity_classes": classes,
    }, indent=2)
    if mode == "text":
        if sensor_text is None:
            raise ValueError("text mode needs sensor_text")
        prompt += "\n\n" + sensor_text
    return prompt
