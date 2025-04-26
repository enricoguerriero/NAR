CONFIG = {
    # directories
    "video_folder": "data/videos",
    "annotation_folder": "data/annotations",
    "token_dir": "data/tokens",
    "feature_dir": "data/features",
    
    # data specific
    "clip_length": 2,  # seconds
    "overlapping": 0.5,  # proportion of overlap
    "frame_per_second": 8,  # target FPS
    "batch_size_feature": 2, # batch size for feature extraction (low because it runs the vlm)
    
    # VLM specific
    "prompt": "Is the baby / mannequin visible? If yes, is the baby receiving ventilation? Is the baby being stimulated? Is the baby receiving suction?",
    "system_message": """
        You are assisting in a medical simulation analysis. A camera is positioned above a table. The simulation involves a mannequin representing a newborn baby, which may or may not be present on the table.
        1  Presence  
            • Is the mannequin clearly visible on the table?  
            - If not visible, there are no treatments to check.
        2  Face (if present)  
            • Locate the face; needed for treatment checks.
        3  Treatments to detect (Stimulation can occur together with other treatments)  
            • Ventilation - mask held on the face.  
            • Stimulation - hands on back / buttocks / trunk with up-down motion.
            • Suction - tube in mouth or nose.  
        Respond concisely and structurally, listing any treatments seen or “No treatment” if none.
        """,
        
    # training specific
    "batch_size": 8,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.01,
    "criterion": "wbce",
    "scheduler": "reduceonplateau",
    "patience": 5,
    "epochs": 20,
    "threshold": 0.5,
    "num_workers": 4
    
}