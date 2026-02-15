# GLM-4.6 Tool Calling Configuration

# API Configuration
API_BASE = "e9b70b19b8a447319b30e4c31e7a9771.ahVbNDDwd997smZC"
MODEL_NAME = "GLM-4.6V-Flash"

# Dataset Configuration
DATASET_PATH = "example_dataset.jsonl"
DATASET_FORMAT = "jsonl"  # Options: "jsonl", "json"

# Output Configuration
OUTPUT_PATH = "results.json"
SAVE_INTERMEDIATE_RESULTS = True

# Processing Configuration
USE_AUTO_PARSING = False  # Set to False if using manual parsing
MAX_ITERATIONS = 10  # Maximum tool-calling iterations per query
BATCH_SIZE = None  # Set to a number to process in batches

# API Request Configuration
MAX_TOKENS = 1024
TEMPERATURE = 1.0
TIMEOUT = 360  # seconds

# Logging Configuration
VERBOSE = True
LOG_FILE = "processing.log"