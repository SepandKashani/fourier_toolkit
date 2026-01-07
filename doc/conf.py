import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # src-layout project structure

project = "Fourier Toolkit"
copyright = "2026, Sepand KASHANI"
author = "Sepand KASHANI"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme options ---------------------------------------------------------------

html_theme = "bizstyle"
html_static_path = ["_static"]

# Extension options -----------------------------------------------------------
mathjax3_config = dict(
    tex=dict(
        macros={
            # Vectors & Matrices --------------------------------------------------
            "bba": r"{\bf{a}}",
            "bbA": r"{\bf{A}}",
            "bbb": r"{\bf{b}}",
            "bbB": r"{\bf{B}}",
            "bbc": r"{\bf{c}}",
            "bbC": r"{\bf{C}}",
            "bbd": r"{\bf{d}}",
            "bbD": r"{\bf{D}}",
            "bbe": r"{\bf{e}}",
            "bbE": r"{\bf{E}}",
            "bbf": r"{\bf{f}}",
            "bbF": r"{\bf{F}}",
            "bbg": r"{\bf{g}}",
            "bbG": r"{\bf{G}}",
            "bbh": r"{\bf{h}}",
            "bbH": r"{\bf{H}}",
            "bbi": r"{\bf{i}}",
            "bbI": r"{\bf{I}}",
            "bbj": r"{\bf{j}}",
            "bbJ": r"{\bf{J}}",
            "bbk": r"{\bf{k}}",
            "bbK": r"{\bf{K}}",
            "bbl": r"{\bf{l}}",
            "bbL": r"{\bf{L}}",
            "bbm": r"{\bf{m}}",
            "bbM": r"{\bf{M}}",
            "bbn": r"{\bf{n}}",
            "bbN": r"{\bf{N}}",
            "bbo": r"{\bf{o}}",
            "bbO": r"{\bf{O}}",
            "bbp": r"{\bf{p}}",
            "bbP": r"{\bf{P}}",
            "bbq": r"{\bf{q}}",
            "bbQ": r"{\bf{Q}}",
            "bbr": r"{\bf{r}}",
            "bbR": r"{\bf{R}}",
            "bbs": r"{\bf{s}}",
            "bbS": r"{\bf{S}}",
            "bbt": r"{\bf{t}}",
            "bbT": r"{\bf{T}}",
            "bbu": r"{\bf{u}}",
            "bbU": r"{\bf{U}}",
            "bbv": r"{\bf{v}}",
            "bbV": r"{\bf{V}}",
            "bbw": r"{\bf{w}}",
            "bbW": r"{\bf{W}}",
            "bbx": r"{\bf{x}}",
            "bbX": r"{\bf{X}}",
            "bby": r"{\bf{y}}",
            "bbY": r"{\bf{Y}}",
            "bbz": r"{\bf{z}}",
            "bbZ": r"{\bf{Z}}",
            "bbZero": r"{\bf{0}}",
            "bbOne": r"{\bf{1}}",
            # Number Spaces ---------------------------------------------------
            "bC": r"{\mathbb{C}}",
            "bE": r"{\mathbb{E}}",
            "bN": r"{\mathbb{N}}",
            "bQ": r"{\mathbb{Q}}",
            "bR": r"{\mathbb{R}}",
            "bS": r"{\mathbb{S}}",
            "bZ": r"{\mathbb{Z}}",
            # Calligraphic Upper-Case Letters ---------------------------------
            "cA": r"{\mathcal{A}}",
            "cB": r"{\mathcal{B}}",
            "cC": r"{\mathcal{C}}",
            "cD": r"{\mathcal{D}}",
            "cE": r"{\mathcal{E}}",
            "cF": r"{\mathcal{F}}",
            "cG": r"{\mathcal{G}}",
            "cH": r"{\mathcal{H}}",
            "cI": r"{\mathcal{I}}",
            "cJ": r"{\mathcal{J}}",
            "cK": r"{\mathcal{K}}",
            "cL": r"{\mathcal{L}}",
            "cM": r"{\mathcal{M}}",
            "cN": r"{\mathcal{N}}",
            "cO": r"{\mathcal{O}}",
            "cP": r"{\mathcal{P}}",
            "cQ": r"{\mathcal{Q}}",
            "cR": r"{\mathcal{R}}",
            "cS": r"{\mathcal{S}}",
            "cT": r"{\mathcal{T}}",
            "cU": r"{\mathcal{U}}",
            "cV": r"{\mathcal{V}}",
            "cW": r"{\mathcal{W}}",
            "cX": r"{\mathcal{X}}",
            "cY": r"{\mathcal{Y}}",
            "cZ": r"{\mathcal{Z}}",
            # Misc ------------------------------------------------------------
            "cj": r"{j}",  # complex j
            "ee": r"{e}",  # Euler's number
            "bigBrack": [r"\left[ #1 \right]", 1],
            "bigCurly": [r"\left\{ #1 \right\}", 1],
            "bigParen": [r"\left( #1 \right)", 1],
            "innerProduct": [r"\langle #1, #2 \rangle", 2],
            "norm": [r"\left\| #1 \right\|_{#2}", 2],
            "real": [r"\Re \bigCurly{#1}", 1],
            "imag": [r"\Im \bigCurly{#1}", 1],
            "abs": [r"\left| #1 \right|", 1],
            "ceil": [r"\left\lceil #1 \right\rceil", 1],
            "floor": [r"\left\lfloor #1 \right\rfloor", 1],
            "kron": r"{\otimes}",
            "krao": r"{\circ}",
            "range": [r"\bigBrack{ #1, #2 }", 2],  # [#1, #2]
            "discreteRange": [  # {#1_1,...,#2_1} x ... x {#1_D,...,#2_D}
                r"\lbrack\!\lbrack #1, #2 \rbrack\!\rbrack",
                2,
            ],
            # Math Operators --------------------------------------------------
            "argmin": r"\operatorname*{arg\,min}",
            "diag": r"\operatorname*{diag}",
            "vecOp": r"\operatorname*{vec}",
            "matOp": r"\operatorname*{mat}",
            "adj": r"\operatorname*{\ast}",
            "hermitian": r"\operatorname*{H}",
            "transpose": r"\operatorname*{T}",
            "prox": r"\operatorname*{prox}",
            "sinc": r"\operatorname*{sinc}",
            "relu": r"\operatorname*{ReLu}",
            "tr": r"\operatorname*{tr}",
            "grad": r"\operatorname*{\nabla}",
            # Fourier-related -------------------------------------------------
            "dftt": r"DFT",
            "dft": r"\operatorname*{\texttt{DFT}}",
            "dtftt": r"DTFT",
            "dtft": r"\operatorname*{\texttt{DTFT}}",
            "fst": r"FS",
            "fs": r"\operatorname*{\texttt{FS}}",
            "ctftt": r"FT",
            "ctft": r"\operatorname*{\texttt{F}}",
            "tuu": r"{\texttt{U} \to \texttt{U}}",
            "tnuu": r"{\texttt{NU} \to \texttt{U}}",
            "tunu": r"{\texttt{U} \to \texttt{NU}}",
            "tnunu": r"{\texttt{NU} \to \texttt{NU}}",
            "circconv": r"\operatorname*{\circledast}",
            "conv": r"\operatorname*{\ast}",
            "epsBandwidth": [r"\cB_{ #1 }", 1],
            # -----------------------------------------------------------------
        },
    )
)

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
