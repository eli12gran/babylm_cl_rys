import os
import sys
import subprocess
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(job):
    try:
        job_input = job["input"]
        command = job_input.get("command", "train")
        
        if command == "train":
            logger.info("Starting training job...")
            
            # Run training script
            result = subprocess.run(
                [sys.executable, "/workspace/train.py"],
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Training completed successfully",
                    "stdout": result.stdout[-1000:],  # Last 1000 chars
                }
            else:
                return {
                    "status": "error",
                    "message": "Training failed",
                    "stderr": result.stderr[-1000:],
                }
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
            
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":
    # For local testing
    test_job = {"input": {"command": "train"}}
    result = handler(test_job)
    print(json.dumps(result, indent=2))