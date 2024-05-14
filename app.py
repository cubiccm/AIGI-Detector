import gradio as gr
import eval

def greet(file):
  result, real_prob = eval.predict(file)
  return [{
    "real": real_prob,
    "fake": 1 - real_prob
  }, result]

demo = gr.Interface(
  fn=greet,
  inputs=["file"],
  outputs=[
    gr.Label(label="Probability"),
    gr.Text(label="Most probable source")
  ],
  title="AI Image Detector"
)

demo.launch(share=True)
