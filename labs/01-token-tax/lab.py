# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 01: The Token Tax
#
# **Act I — The Landscape** | CPU only | ~30 minutes
#
# Every token a model processes costs compute. But not all text tokenizes equally.
# In this lab you'll load 4 real BPE tokenizers, feed them the same text in 5 languages,
# and measure the **token tax** — how many more tokens a non-English user pays
# for the same semantic content.
#
# **Aha moment:** You'll see that GPT-2's tokenizer charges Hindi ~4-5x more tokens
# than English — and that modern tokenizers (o200k) close the gap to ~1.3x.

# %% [markdown]
# ## Setup

# %%
# Install microscale if running in Colab
import subprocess
import sys

try:
    import microscale
except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/microscale-academy/labs.git",
        ]
    )

from microscale import apply_style, device_summary, show
from microscale.viz import heatmap

apply_style()
print(device_summary())

# %%
import numpy as np
import tiktoken
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ## Step 1: Load the Tokenizers
#
# We compare 4 BPE encodings spanning the history of OpenAI's tokenizers:
#
# | Encoding | Used By | Vocab Size |
# |----------|---------|------------|
# | `gpt2` | GPT-2, GPT-3 | 50,257 |
# | `p50k_base` | Codex, text-davinci-003 | 50,281 |
# | `cl100k_base` | GPT-3.5, GPT-4 | 100,256 |
# | `o200k_base` | GPT-4o, o1 | 200,019 |

# %%
ENCODINGS = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "p50k_base": tiktoken.get_encoding("p50k_base"),
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
}

table = Table(title="Tokenizer Vocabulary Sizes")
table.add_column("Encoding", style="bold")
table.add_column("Vocab Size", justify="right")
for name, enc in ENCODINGS.items():
    table.add_row(name, f"{enc.n_vocab:,}")
console.print(table)

# %% [markdown]
# ## Step 2: Define Multi-Language Test Corpus
#
# We need semantically equivalent text across languages. These are short passages
# expressing the same idea — how a language model processes input.

# %%
CORPUS = {
    "English": (
        "The small language model processed the input sequence in twelve milliseconds. "
        "Each token was mapped to an embedding vector before the attention mechanism "
        "computed the weighted relationships between all positions in the context window."
    ),
    "Hindi": (
        "छोटे भाषा मॉडल ने इनपुट अनुक्रम को बारह मिलीसेकंड में संसाधित किया। "
        "ध्यान तंत्र द्वारा संदर्भ विंडो में सभी स्थितियों के बीच भारित संबंधों "
        "की गणना करने से पहले प्रत्येक टोकन को एक एम्बेडिंग वेक्टर में मैप किया गया।"
    ),
    "Japanese": (
        "小型言語モデルは入力シーケンスを12ミリ秒で処理しました。"
        "アテンションメカニズムがコンテキストウィンドウ内のすべての位置間の"
        "重み付き関係を計算する前に、各トークンは埋め込みベクトルにマッピングされました。"
    ),
    "Arabic": (
        "قام نموذج اللغة الصغير بمعالجة تسلسل الإدخال في اثني عشر ملي ثانية. "
        "تم تعيين كل رمز إلى متجه تضمين قبل أن تحسب آلية الانتباه "
        "العلاقات الموزونة بين جميع المواضع في نافذة السياق."
    ),
    "Python": (
        "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "    residual = x\n"
        "    x = self.norm(x)\n"
        "    x = self.attention(x)\n"
        "    x = residual + x\n"
        "    residual = x\n"
        "    x = self.norm(x)\n"
        "    x = self.ffn(x)\n"
        "    return residual + x\n"
    ),
}

# %% [markdown]
# ## Step 3: Compute Token Counts
#
# For each (tokenizer, language) pair, count the tokens produced.

# %%
languages = list(CORPUS.keys())
encodings = list(ENCODINGS.keys())

token_counts = np.zeros((len(languages), len(encodings)), dtype=int)

for i, lang in enumerate(languages):
    for j, enc_name in enumerate(encodings):
        tokens = ENCODINGS[enc_name].encode(CORPUS[lang])
        token_counts[i, j] = len(tokens)

table = Table(title="Token Counts by Language and Encoding")
table.add_column("Language", style="bold")
for enc_name in encodings:
    table.add_column(enc_name, justify="right")

for i, lang in enumerate(languages):
    table.add_row(lang, *[str(token_counts[i, j]) for j in range(len(encodings))])

console.print(table)

# %% [markdown]
# ## Step 4: Compute the Token Tax
#
# The **token tax** is the ratio of tokens for a given language vs English.
# A tax of 3.0x means you need 3x more tokens to express the same idea —
# which means 3x the inference cost, 3x the latency, and 1/3 the effective
# context window.

# %%
english_idx = languages.index("English")
token_tax = token_counts / token_counts[english_idx, :]

table = Table(title="Token Tax (relative to English = 1.0x)")
table.add_column("Language", style="bold")
for enc_name in encodings:
    table.add_column(enc_name, justify="right")

for i, lang in enumerate(languages):
    row = []
    for j in range(len(encodings)):
        tax = token_tax[i, j]
        color = "green" if tax <= 1.5 else "yellow" if tax <= 3.0 else "red"
        row.append(f"[{color}]{tax:.2f}x[/{color}]")
    table.add_row(lang, *row)

console.print(table)

# %% [markdown]
# ## Step 5: Visualize as a Heatmap
#
# A heatmap makes the fairness gap immediately visible.

# %%
fig = heatmap(
    token_tax,
    title="Token Tax: How Many More Tokens Than English?",
    xlabel="Tokenizer Encoding",
    ylabel="Language",
    xticklabels=encodings,
    yticklabels=languages,
    cmap="YlOrRd",
    figsize=(10, 5),
)
show(fig, filename="01-token-tax-heatmap.png")

# %% [markdown]
# ## Step 6: Interactive Exploration (Plotly)
#
# For a richer view, here's an interactive heatmap you can hover over.

# %%
import plotly.graph_objects as go

fig_interactive = go.Figure(
    data=go.Heatmap(
        z=token_tax,
        x=encodings,
        y=languages,
        colorscale="YlOrRd",
        text=[[f"{v:.2f}x" for v in row] for row in token_tax],
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Language: %{y}<br>Encoding: %{x}<br>Tax: %{text}<extra></extra>",
    )
)
fig_interactive.update_layout(
    title="Token Tax: Interactive Heatmap",
    xaxis_title="Tokenizer Encoding",
    yaxis_title="Language",
    width=700,
    height=400,
)

from microscale.env import is_notebook as _is_nb

if _is_nb():
    fig_interactive.show()
else:
    from microscale.viz import _output_dir

    path = _output_dir() / "01-token-tax-interactive.html"
    fig_interactive.write_html(str(path), include_plotlyjs=True)
    console.print(f"  [dim]Interactive heatmap saved:[/dim] {path}")

# %% [markdown]
# ## Key Takeaways
#
# **What you measured:**
# - GPT-2's tokenizer imposes a ~4-5x tax on Hindi and Arabic
# - `o200k_base` (200k vocab) cuts that to ~1.3-1.5x
# - Python code tokenizes efficiently across all encodings (ASCII-heavy)
# - Larger vocabularies help multilingual equity but increase embedding table size
#
# **Why it matters for SLMs:**
# A small model with a 32k vocabulary pays an even higher token tax than GPT-2.
# When deploying SLMs for multilingual use, the tokenizer choice directly
# impacts effective context length, latency, and cost.
#
# ## Artifact
#
# Check your `outputs/` directory for:
# - `01-token-tax-heatmap.png` — static heatmap
# - `01-token-tax-interactive.html` — interactive Plotly heatmap
