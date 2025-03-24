import dspy
from dspy_llamacpp import AutoLlamaCpp
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the model path from the environment
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH is not set in the environment")

# Instantiate the LM via AutoLlamaCpp
llama_lm = AutoLlamaCpp(
    MODEL_PATH,
    port=8080,
    server_options={"--verbosity": -1},
    temperature=0.5,
)

questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What is the largest planet in our solar system?",
    "Who wrote the play 'Romeo and Juliet'?",
    "What is the boiling point of water?",
]

program = dspy.Predict("question -> answer: str")


for q in questions:
    print(f"Q: {q}")
    answer = program(question=q)
    print(f"A: {answer}")
    print("-" * 20)
