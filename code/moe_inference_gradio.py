import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import re
import gradio as gr

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
     )
}

class Args:
    def __init__(self):
        self.bert_rounter = "/path/to/bert/rounter_model"
        self.bert_model = "/path/to/product_predict/model"
        self.com_plan_model = "/path/to/plan/model"
        self.code_model = "/path/to/code/model"
        self.context_size = 32768
        self.max_gen_len = 30000
        self.cache_dir = "./cache"
        self.temperature = 0.6
        self.top_p = 0.9

def get_label_map():
    return {
        0: "The main product is CH3CH2OH",
        1: "The main product is C2H4",
        2: "The main product is HCOOH/HCOO-",
        3: "The main product is C2+",
        4: "The main product is CH3OH",
        5: "The main product is CH4",
        6: "The main product is CO"}
    
def load_BERT_Rounter(model_path, query):
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
    model = transformers.BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.eval()
    label_list = [f'Label_{i}' for i in range(model.config.num_labels)]
    inputs = tokenizer(query, return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = label_list[predicted_label_id]
        confidence = probabilities[0][predicted_label_id].item()
    return predicted_label

def load_bert(model_name):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def run_bert(text, model, tokenizer):
    label_map = get_label_map()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    product_label = label_map[predictions.item()]
    return product_label

def build_generator(model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer)
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        return out
    return response

def load_llama_model(model_path, device, context_size, cache_dir):
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.to(device)
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        model_max_length=context_size if context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    model.eval()

    return model, tokenizer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_label = load_BERT_Rounter(args.bert_rounter, args.question)

    if predicted_label == 'Label_0':
        bert_model, bert_tokenizer = load_bert(args.bert_model)
        product_label = run_bert(args.question, bert_model, bert_tokenizer)
        return f"Product Prediction Result:\n{product_label}"

    elif predicted_label == 'Label_1':
        plan_model, plan_tokenizer = load_llama_model(args.com_plan_model, device, args.context_size, args.cache_dir)
        plan_response = build_generator(plan_model, plan_tokenizer, temperature=args.temperature, top_p=args.top_p, max_gen_len=args.max_gen_len, use_cache=True)
        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        prompt = prompt_no_input.format_map({"instruction": args.question})
        output = plan_response(prompt=prompt)
        return f"Computational Planning Result:\n{output}"

    else:
        code_model, code_tokenizer = load_llama_model(args.code_model, device, args.context_size, args.cache_dir)
        code_response = build_generator(code_model, code_tokenizer, temperature=args.temperature, top_p=args.top_p, max_gen_len=args.max_gen_len, use_cache=True)
        prompt_no_input = PROMPT_DICT["prompt_no_input_llama2"]
        prompt = prompt_no_input.format_map({"instruction": args.question})
        output = code_response(prompt=prompt)
        return f"Code Generation Result:\n{output}"

def process_question(question):
    args = Args()
    args.question = question
    try:
        result = main(args)
        return result
    except Exception as e:
        return f"Error occurred: {str(e)}"

def create_demo():
    iface = gr.Interface(
        fn=process_question,
        inputs=gr.Textbox(
            lines=3, 
            placeholder="Enter your CO2RR related question here...",
            label="Question"
        ),
        outputs=gr.Textbox(
            lines=10,
            label="Response"
        ),
        title="CO2RR Assistant",
        description="Ask questions about CO2 reduction reaction (CO2RR), including product prediction, computational planning, and code generation.",
        examples=[
            ["What is the main product of CO2RR on Cu(100) surface at -0.9V vs. RHE?"],
            ["Simulate the CO2 reduction reaction (CO2RR) to produce CH4 (methane) on a Cs-Lu alloy (111) surface."],
            ["Generate VASP input files for CO2RR simulation on Cu(111) surface"]
        ]
    )
    return iface

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=None
    )