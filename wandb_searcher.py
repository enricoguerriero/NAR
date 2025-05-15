import wandb

api = wandb.Api()
runs = api.runs("guerriero/NewbornActivityRecognition")

for run in runs:
    if run.config.get("test_type") == "0-shot":
        print(run.name, run.config)
