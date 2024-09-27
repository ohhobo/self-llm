from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

with st.sidebar:
    st.markdown("## Qwen 2.5 LLM")
    max_length = st.slider("Max length", 0, 8192, 512,step = 1)
    
st.title("💬 Qwen2.5 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

model_name_or_path = "./models/qwen/Qwen2___5-7B-Instruct"

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16,device_map='auto')
    return tokenizer, model

tokenzier,model = get_model()

if st.session_state.get('message') is None:
    st.session_state['message'] = [{"role":"assistant","content":"你好,我是Qwen2.5，请问有什么可以帮助你的吗？"}]
    
for msg in st.session_state['message']:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state['message'].append({"role":"user","content":prompt})
    input_ids = tokenzier.apply_chat_template(st.session_state['message'],tokenize=False,add_generation_prompt=True)
    model_inputs = tokenzier([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=max_length)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenzier.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.session_state['message'].append({"role":"assistant","content":response})
    st.chat_message("assistant").write(response)
    
    
