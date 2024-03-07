import os,re
from run.glm_infer import CHATGLM
from run.tagger_infer import WD_TAG
import gradio as gr

class GradioWdGlmImgsanalys():
    def __init__(self) -> None:
        self.init_model()
        pass

    def init_model(self):
        self.chatglm = CHATGLM()
        self.wd_tag = WD_TAG()

    def run_post_res(self,image):
        tag = self.wd_tag.infer(image)
        res = self.chatglm.predict(tag)
        pos_res = re.findall(r'[\u4e00-\u9fff\uf900-\ufaff\uff01-\uffef]+', res)

        return ''.join(pos_res)

    def del_res(self,a,b):
        return gr.Image.update(value=None),gr.Text.update(value="")

    def init_ui(self):
        with gr.Blocks(title="Gradio-WD14-GLM-ImageAnalysis") as demo:
            gr.Markdown("""
                        <p align="center"><img src='file\imgs\mt_logo.png' alt='image One' style="height: 200px"/><p>""")
            gr.Markdown("""<center><font size=8>Gradio-WD14-GLM-ImageAnalysis</center>""")
            gr.Markdown(
                """\
                        <center><font size=3>该项目基于gradio、wd1.4、chatglm杂交组合，变相实现图像分析的功能(自己用于练习Gradio，没有额外价值)。</center>""")
            gr.Markdown("""\
                        <center><font size=4>
                        Gradio <a href="https://www.gradio.app/"> ✨✨ </a> ｜ 
                        ChatGLM-6B <a href="https://github.com/THUDM/ChatGLM-6B"> ✨✨ </a>&nbsp ｜ 
                        WD14-Tagger <a href="https://github.com/toriato/stable-diffusion-webui-wd14-tagger"> ✨✨</a>&nbsp 
                        </center>""")
            
            with gr.Row():
                input_image = gr.Image(label="图片",scale=5)
                with gr.Column():
                    ok_button = gr.Button("确认",scale=1)
                    del_button = gr.Button("清空",scale=1)
            output_txt = gr.Text(label="结果",lines=8,max_lines=8)

            ok_button.click(self.run_post_res,inputs=input_image,outputs=output_txt)
            del_button.click(self.del_res,inputs=[input_image,output_txt],outputs=[input_image,output_txt])

            return demo

if __name__ == "__main__":
    gwgis = GradioWdGlmImgsanalys()
    demo = gwgis.init_ui()
    demo.queue(150).launch(server_port=8080,share=False,show_api=False)
