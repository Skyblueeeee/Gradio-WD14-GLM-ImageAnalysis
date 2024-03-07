# 导入所需的库
from transformers import AutoModel, AutoTokenizer  # 使用transformers库中的AutoModel和AutoTokenizer
import gradio as gr  # 导入gradio库
import mdtex2html  # 导入mdtex2html库
import os  # 导入os库

from utils import load_model_on_gpus  # 从utils模块中导入load_model_on_gpus函数

# 加载预训练的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("model/chatglm-6b-int4", trust_remote_code=True)  # 使用指定的tokenizer
model = AutoModel.from_pretrained("model/chatglm-6b-int4", trust_remote_code=True).half().cuda()  # 使用指定的model，并将其转换为半精度浮点数并移动到GPU上
model = model.eval()  # 设置model为评估模式

# 重写Chatbot类的postprocess方法
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),  # 将message转换为HTML格式
            None if response is None else mdtex2html.convert(response),  # 将response转换为HTML格式
        )
    return y

gr.Chatbot.postprocess = postprocess  # 将postprocess方法设置为Chatbot类的postprocess方法

# 解析文本
def parse_text(text):
    """从https://github.com/GaiZhenbiao/ChuanhuChatGPT/复制而来"""
    lines = text.split("\n")  # 将文本按换行符分割成行
    lines = [line for line in lines if line != ""]  # 去除空行
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'  # 替换为pre和code标签
            else:
                lines[i] = f'<br></code></pre>'  # 替换为br、/code和/pre标签
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")  # 替换`为\`
                    line = line.replace("<", "&lt;")  # 替换<为&lt;
                    line = line.replace(">", "&gt;")  # 替换>为&gt;
                    line = line.replace(" ", "&nbsp;")  # 替换空格为&nbsp;
                    line = line.replace("*", "&ast;")  # 替换*为&ast;
                    line = line.replace("_", "&lowbar;")  # 替换_为&lowbar;
                    line = line.replace("-", "&#45;")  # 替换-为&#45;
                    line = line.replace(".", "&#46;")  # 替换.为&#46;
                    line = line.replace("!", "&#33;")  # 替换!为&#33;
                    line = line.replace("(", "&#40;")  # 替换(为&#40;
                    line = line.replace(")", "&#41;")  # 替换)为&#41;
                    line = line.replace("$", "&#36;")  # 替换$为&#36;
                lines[i] = "<br>"+line  # 添加<br>标签
    text = "".join(lines)  # 将处理后的行拼接成文本
    return text


# 预测函数
# def predict(input, chatbot, max_length, top_p, temperature, history):
#     chatbot.append((parse_text(input), ""))  # 将用户输入添加到chatbot中
#     for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
#                                                temperature=temperature):
#         chatbot[-1] = (parse_text(input), parse_text(response))  # 更新chatbot的最后一条对话
#         print(chatbot,response)  # 打印chatbot和response
#         yield chatbot, history  # 生成chatbot和history



MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
 
first_ans = ('','')
second_ans = ('','')
 
def predict(input, max_length, top_p, temperature, history=None):
    global first_ans
    global second_ans
    if history is None:
        history = []
    for response, history in model.stream_chat(tokenizer, input, max_length=max_length, top_p=top_p,
    temperature=temperature, history=(history if len(history) <= 4 else [first_ans] + [second_ans] + history[-3:])):
        if len(history) <= 1 :
            first_ans = history[0]
        if len(history) <= 2 and len(history) > 1 :
            second_ans = history[1]
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="用户：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))

        yield [history] + updates



# 重置用户输入
def reset_user_input():
    return gr.update(value='')

# 重置状态
def reset_state():
    return [], []

# 创建交互界面
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")  # 在界面中添加HTML标题

    chatbot = gr.Chatbot()  # 创建Chatbot组件
    with gr.Row():  # 创建一行组件
        with gr.Column(scale=4):  # 创建一个占据4/5宽度的列
            with gr.Column(scale=12):  # 创建一个占据12/12宽度的列
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)  # 创建文本框组件
            with gr.Column(min_width=32, scale=1):  # 创建一个最小宽度为32，占据1/5宽度的列
                submitBtn = gr.Button("Submit", variant="primary")  # 创建提交按钮
        with gr.Column(scale=1):  # 创建一个占据1/5宽度的列
            emptyBtn = gr.Button("Clear History")  # 创建清空历史按钮
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)  # 创建最大长度滑块
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)  # 创建Top P滑块
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)  # 创建Temperature滑块

    history = gr.State([])  # 创建一个空的状态

    # 绑定按钮的点击事件
    submitBtn.click(predict, [user_input, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)  # 绑定提交按钮的点击事件
    submitBtn.click(reset_user_input, [], [user_input])  # 绑定提交按钮的点击事件
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)  # 绑定清空历史按钮的点击事件

demo.queue().launch(share=True, inbrowser=True)  # 启动交互界面并在浏览器中打开