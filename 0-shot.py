'''
Script to test the NAR model with 0-shot prompting.
(Hopefully) lightweight and easy.
'''
import os
from argparse import ArgumentParser
from utils import setup_logging, setup_wandb, load_model
from config import CONFIG
from data.token_dataset import TokenDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline
)
import torch
import json
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

def main():
    
    # Parse command line arguments
    parser = ArgumentParser(description="Test NAR model with 0-shot prompting.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to test.")
    parser.add_argument("--export_tokens", action="store_true", help="Export tokens to a file.")
    
    args = parser.parse_args()
    model_name = args.model_name
    export_tokens = args.export_tokens
    
    # Set up logging
    logger = setup_logging(model_name)
    logger.info("Starting 0 shot test script.")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Export tokens: {export_tokens}")
    
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hf_token)
    
    # Set up Weights & Biases
    config = {
        "model_name": model_name,
        "test_type": "0-shot",
        "data": "test data",
    }
    wandb_run = setup_wandb(model_name, config)
    logger.info("Weights & Biases setup complete.")
    
    # Load the model
    model = load_model(model_name, None)
    logger.info(f"Model {model_name} loaded successfully.")
    logger.info(f"Model device: {model.device}")
    
    # Export tokens
    if export_tokens:
        token_dir = os.path.join("data/tokens", model_name, "0-shot", "trial")
        os.makedirs(token_dir, exist_ok=True)
        logger.info(f"Exporting tokens to {token_dir}.")
        
        model.export_tokens(video_folder = os.path.join(CONFIG["video_folder"], "test"),
                                annotation_folder = os.path.join(CONFIG["annotation_folder"], "test"),
                                output_folder = token_dir,
                                clip_length = CONFIG["clip_length_0s"],
                                overlapping = CONFIG["overlapping_0s"],
                                frame_per_second = CONFIG["frame_per_second_0s"],
                                prompt = CONFIG["prompt_0s"],
                                system_message = CONFIG["system_message_0s"],
                                logger = logger)
        logger.info("Tokens exported successfully.")
        token_dir = os.path.join(token_dir, f'{CONFIG["clip_length_0s"]}sec_{CONFIG["frame_per_second_0s"]}fps')
    else:
        logger.info("Exporting tokens is skipped.")
        token_dir = os.path.join("data/tokens", model_name, "0-shot", "trial", f"{CONFIG['clip_length_0s']}sec_{CONFIG['frame_per_second_0s']}fps")
    
    token_dataset = TokenDataset(token_dir)
    logger.info(f"Token dataset loaded from {token_dir}.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Judge model hardcoded by now
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    logger.info(f"Loading judge model {MODEL_NAME}.")
    judge_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    
    logger.info("Setting up judge pipeline.")
    judge_pipe = TextGenerationPipeline(
        model=judge_model,
        tokenizer=judge_tokenizer,
        task="text-generation"
    )
    
    LABELS = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    JUDGE_PROMPT = """[INST]
        You are given an image caption. For each of these labels: {labels}, output a JSON object with keys as label names and boolean values indicating presence.
        Respond with exactly the JSON object and no other text.
        Caption: "{caption}"
        [/INST]"""
    
    def judge_caption(caption: str) -> dict:
        prompt = JUDGE_PROMPT.format(
            labels=", ".join(LABELS),
            caption=caption.replace('"', '\\"')
        )
        # Generate
        out = judge_pipe(
            prompt,
            max_new_tokens=64,
            do_sample=False,        # deterministic
            eos_token_id=judge_tokenizer.eos_token_id
        )[0]["generated_text"]

        # Extract and parse the JSON
        try:
            # Find the first '{' to catch leading whitespace
            json_str = out[out.index("{"):out.rindex("}")+1]
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from model output: {out}") from e
    
    TP = [0, 0, 0, 0]
    FP = [0, 0, 0, 0]
    TN = [0, 0, 0, 0]
    FN = [0, 0, 0, 0]
    total_counter = 0
    # Test the model
    logger.info("Starting model testing.")
    for i, clip in tqdm(enumerate(token_dataset), total=len(token_dataset), desc="Testing clips", unit="clip"):
        
        logger.info("Inspecting clip...")
        pixel_values = clip.get("pixel_values", None)
        pixel_values_videos = clip.get("pixel_values_videos", None)
        input_ids = clip.get("input_ids", None)
        attention_mask = clip.get("attention_mask", None)
        
        logger.info("Generating answer for clip.")
        answer = model.generate_answer(inputs = clip,
                                       max_new_tokens = 128,
                                       do_sample = False)
        
        logger.info("Judge model judging...")
        # Send the caption to the judge model
        predicted_labels = judge_caption(answer)
        
        # Log real labels, answer, and predicted labels
        real_labels = clip["labels"].tolist()
        logger.info(f"Clip {i}:")
        logger.info(f"Real labels: {real_labels}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Predicted labels: {predicted_labels}")
        
        # check if the predicted labels match the real labels
        # before that, convert the predicted labels to a list
        predicted_labels_list = [int(predicted_labels.get(label, False)) for label in LABELS]
        logger.info(f"Predicted labels list: {predicted_labels_list}")

        for j in range(4):
            if predicted_labels_list[j] == real_labels[j]:
                if predicted_labels_list[j] == 1:
                    TP[j] += 1
                else:
                    TN[j] += 1
            else:
                if predicted_labels_list[j] == 1:
                    FP[j] += 1
                else:
                    FN[j] += 1
        total_counter += 1
    
    logger.info("Model testing complete.")
    logger.info(f"Total clips: {total_counter}")
    logger.info(f"TP: {TP}")
    logger.info(f"FP: {FP}")
    logger.info(f"TN: {TN}")
    logger.info(f"FN: {FN}")
    
    # Compute metrics per class
    metrics = {}
    for i, label in enumerate(LABELS):
        TP_i = TP[i]
        FP_i = FP[i]
        TN_i = TN[i]
        FN_i = FN[i]
        
        accuracy = (TP_i + TN_i) / total_counter if total_counter > 0 else 0
        precision = TP_i / (TP_i + FP_i) if (TP_i + FP_i) > 0 else 0
        recall = TP_i / (TP_i + FN_i) if (TP_i + FN_i) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            "TP": TP_i,
            "FP": FP_i,
            "TN": TN_i,
            "FN": FN_i,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }
        
    # log on wandb
    wandb_run.log({"metrics": metrics})
    wandb_run.finish()
    logger.info("Metrics logged to Weights & Biases.")
    logger.info("0-shot test script complete.")
    logger.info("Bye bye!")
    
if __name__ == "__main__":
    main()