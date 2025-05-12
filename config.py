CONFIG = {
    # directories
    "video_folder": "data/videos",
    "annotation_folder": "data/annotations",
    "token_dir": "data/tokens",
    "feature_dir": "data/features",
    
    # data specific
    "clip_length": 2,  # seconds
    "overlapping": 0.5,  # proportion of overlap
    "frame_per_second": 4,  # target FPS
    "batch_size_feature": 1, # batch size for feature extraction (low because it runs the vlm)
    
    # VLM specific
    "prompt": "Is the baby / mannequin visible? If yes, is the baby receiving ventilation? Is the baby being stimulated? Is the baby receiving suction? Ventilation and suction can not be performed at the same time.",
    "system_message": """
        You are assisting in a medical simulation analysis. A camera is positioned above a table. The simulation involves a mannequin representing a newborn baby, which may or may not be present on the table.

        Your tasks are as follows:

        1. Determine Presence
        - Check if the mannequin (baby) is visible and present on the table.
        - If not present or not visible, no treatment is being performed.
        - If present, continue to the next steps.

        2. Identify the Mannequin's Face
        - Locate the face of the mannequin. This is the key area for identifying some treatments.

        3. Detect Medical Treatments
        If the mannequin is present, identify whether the following treatments are being performed. These treatments can occur individually or at the same time:

        - Ventilation:
            - A healthworker is holding a ventilation mask over the mannequin's face.
            - This means ventilation is being administered.

        - Suction:
            - A tube is inserted into the mannequin's mouth or nose.
            - This means suction is being performed.

        - Stimulation:
            - A healthworker is applying stimulation to the mannequin's back, buttocks (nates), or trunk.
            - Stimulation is indicated by:
            - Hands placed on one of these areas
            - Up-and-down repetitive hand movement

        Repeat: If the mannequin is not visible, no treatment is being performed.

        Respond clearly based on what is visible in the image. Use concise and structured output when possible.
        """,
        
    # training specific
    "batch_size": 8,
    "optimizer": "adamw",
    "learning_rate": 6.515e-05,
    "momentum": 0.9,
    "weight_decay": 1.413e-06,
    "criterion": "wbce",
    "scheduler": "reduceonplateau",
    "scheduler_patience": 1,
    "patience": 7,
    "epochs": 15,
    "threshold": 0.3,
    "num_workers": 0,
    "freezing_condition": "lora",
    
    # 0 shot specific
    "prompt_0s": "Describe the scene. Be explicit about the presence of the mannequin and the treatments being performed, if any. Check for ventilation, stimulation, and suction.",
    "system_message_0s": """
        You are assisting in a medical simulation analysis. A camera is positioned above a table. The simulation involves a mannequin representing a newborn baby, which may or may not be present on the table.

        Your tasks are as follows:

        1. Determine Presence
        - Check if the mannequin (baby) is visible and present on the table.
        - If not present or not visible, no treatment is being performed.
        - If present, continue to the next steps.

        2. Identify the Mannequin's Face
        - Locate the face of the mannequin. This is the key area for identifying some treatments.

        3. Detect Medical Treatments
        If the mannequin is present, identify whether the following treatments are being performed. These treatments can occur individually or at the same time:

        - Ventilation:
            - A healthworker is holding a ventilation mask over the mannequin's face.
            - This means ventilation is being administered.

        - Suction:
            - A tube is inserted into the mannequin's mouth or nose.
            - This means suction is being performed.

        - Stimulation:
            - A healthworker is applying stimulation to the mannequin's back, buttocks (nates), or trunk.
            - Stimulation is indicated by:
            - Hands placed on one of these areas
            - Up-and-down repetitive hand movement

        Repeat: If the mannequin is not visible, no treatment is being performed.

        Respond clearly based on what is visible in the image. Use concise and structured output when possible.
        """,
    "clip_length_0s": 2,  # seconds
    "overlapping_0s": 0.5,  # proportion of overlap
    "frame_per_second_0s": 4,  # target FPS
    "batch_size_feature_0s": 1, # batch size for feature extraction (low because it runs the vlm)
    
    
}