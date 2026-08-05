from .utils.call_llm_prompt_v1 import Prompt_v1
from .utils.call_llm_prompt_v2 import Prompt_v2
from .utils.call_llm_prompt_v3 import Prompt_v3
from pathlib import Path

prompt_v1 = Prompt_v1()
prompt_v2 = Prompt_v2()
prompt_v3 = Prompt_v3()

data_dir = Path("src/assignment/data")
data_dir.mkdir(parents=True, exist_ok=True)

for filename in [
    "prompt_v1.json",
    "prompt_v2.json",
    "prompt_v3.json",
]:
    (data_dir / filename).touch(exist_ok=True)

with open("src/assignment/data/prompt_v1.json", "r") as f:
    v1_data = f.read()
with open("src/assignment/data/prompt_v2.json", "r") as f:
    v2_data = f.read()
with open("src/assignment/data/prompt_v3.json", "r") as f:
    v3_data = f.read()


if v1_data:
    v1_pass_percentage, v1_compare = prompt_v1.check_result()
else:
    print("Version 1 Processing")
    prompt_v1.process()
    v1_pass_percentage, v1_compare = prompt_v1.check_result()

print(f"Version 1 Pass Percentage :- {v1_pass_percentage}")


if v2_data:
    v2_pass_percentage, v2_compare = prompt_v2.check_result()
else:
    print("Version 2 Processing")
    prompt_v2.process()
    v2_pass_percentage, v2_compare = prompt_v2.check_result()

print(f"Version 2 Pass Percentage :- {v2_pass_percentage}")


if v3_data:
    v3_pass_percentage, v3_compare = prompt_v3.check_result()
else:
    print("Version 3 Processing")
    prompt_v3.process()
    v3_pass_percentage, v3_compare = prompt_v3.check_result()

print(f"Version 3 Pass Percentage :- {v3_pass_percentage}")
print(v3_compare)
