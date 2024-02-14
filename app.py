import gradio as gr
from load_model import predict

def run(text):
    output = str(round(predict(text), 1))
    return output

with gr.Blocks() as demo:
    gr.Markdown("Enter the sentence you want to analyze.")
    with gr.Row():
        inp = gr.Textbox(label="Sentence")
        out = gr.Textbox(label="Score")
    btn = gr.Button("Run")
    btn.click(fn=run, inputs=inp, outputs=out)

demo.launch()