import gradio as gr

def my_function(input_text, option1, option2):
    output_text = f"Input Text: {input_text}, Option 1: {option1}, Option 2: {option2}"
    return output_text

ui = gr.Interface(
    fn=my_function,
    inputs=[
        gr.Textbox(default="Enter Text Here", label="Input Text"),
        gr.Block([
            gr.Checkbox(default=False, label="Option 1", key="option1"),
            gr.Checkbox(default=False, label="Option 2", key="option2"),
        ], label="Options"),
    ],
    outputs=gr.Textbox(),
    live=True,
    layout="vertical",
    title="Gradio Demo",
    description="This is a simple Gradio demo with text input, checkboxes, and output.",
)

ui.launch()
