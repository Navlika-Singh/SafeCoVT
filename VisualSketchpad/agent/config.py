import os

# set up the agent
MAX_REPLY = 10

# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = ''
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={"cache_seed": None, "config_list": [{"model": "gpt-3.5-turbo", "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}


# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = "https://ebe5430a3b1e4a6771.gradio.live"
GROUNDING_DINO_ADDRESS = "https://36ae423d027219fcc2.gradio.live"
DEPTH_ANYTHING_ADDRESS = "https://70c3501c0a193dc255.gradio.live"
