from LLM_langchain import Qwen2_5_LLM

llm = Qwen2_5_LLM(model_name_or_path="./models/qwen/Qwen2___5-7B-Instruct")
print(llm("你好"))
