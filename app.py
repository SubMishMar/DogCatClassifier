import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('cat_dog_model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Dog Cat Classifier"
description = "A Dog Cat Classifier trained on Dog and Cat images downloaded from internet with fastai. Created as a demo for Gradio and HuggingFace Spaces."
#article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
#examples = ['siamese.jpg']
#interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()
