from flask import Flask, render_template, request
from pathlib import Path
from groq import Groq
from bs4 import BeautifulSoup
import requests
import re
import shutil

app = Flask(__name__)

script_dir = Path(__file__).parent
resume_root = script_dir / "resumes"
resume_root.mkdir(exist_ok=True)
resume_folder = resume_root / "1"
resume_folder.mkdir(exist_ok=True)

def download_resume_from_url(url: str) -> Path:
    # clear old files
    for item in resume_folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # download html
    response = requests.get(url)
    response.raise_for_status()

    file_path = resume_folder / "page.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return file_path

def html_text(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text().split())

@app.route("/", methods=["GET", "POST"])
def index():
    jd_result = None
    if request.method == "POST":
        resume_url = request.form.get("resume_url")
        if resume_url:
            # download + parse
            resume_file = download_resume_from_url(resume_url)
            raw_text = html_text(resume_file)

            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            system_message = (
                "You are a professional resume writer with 10 years of experience. "
                "These details must be present at the top: "
                "Job Title: <job title>, Company Name: <company name>, Location: <location>. "
                "Write a polished job description based on the provided text but do not omit any keywords. "
                "Return the complete job description verbatim, preserving every word, using bullet points. "
                "Exclude other details such as recommended jobs or unrelated content."
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Extract the job description from this:\n\n{raw_text}"}
            ]

            chat = groq_client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=messages,
                temperature=0.2,
                top_p=0.9,
                stream=False,
            )

            reply = chat.choices[0].message.content
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            jd_result = reply

    return render_template("index.html", jd_result=jd_result)

if __name__ == "__main__":
    app.run(debug=True)
