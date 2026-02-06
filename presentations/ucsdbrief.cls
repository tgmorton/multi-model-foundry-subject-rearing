% This material is declared a work of the United States Government and is not
% subject to copyright protection in the United States.
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% ------------------------------------------------------------------------------
% Original Author: David Woodburn
% Modified for UCSD branding: 2025/10/16
% Description: UCSD presentation template
% ------------------------------------------------------------------------------
% The dimensions of the text box are
%
%   Columns |  Width  | Height
%   ------- | ------- | ------
%      1    | 14.0 cm | 6.5 cm
%      2    |  6.5 cm | 6.5 cm

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ucsdbrief}[2025/10/16 UC San Diego Presentation]
\def\@version{20251016}

% Process options.
\newif\ifstandard\standardfalse
\newif\ifdefineul\defineultrue
\def\@theme{0} % default color theme
\DeclareOption{standard}{\standardtrue}
\DeclareOption{wide}{\standardfalse}
\DeclareOption{noul}{\defineulfalse}
\DeclareOption{yellow}{\def\@theme{1}}
\DeclareOption{orange}{\def\@theme{2}}
\DeclareOption{red}{\def\@theme{3}}
\DeclareOption{blue}{\def\@theme{4}}
\DeclareOption{purple}{\def\@theme{5}}
\DeclareOption{green}{\def\@theme{6}}
\DeclareOption*{\PackageWarning{ucsdbrief}{Unknown '\CurrentOption'}}
\ProcessOptions\relax
\ifstandard
    \LoadClass[10pt]{article}
\else
    \LoadClass[12pt]{article}
\fi

% Packages
\RequirePackage[T1]{fontenc} % use 8-bit fonts
\RequirePackage{geometry} % page dimensions and margins
\RequirePackage{fancyhdr} % custom headers
\RequirePackage{float} % custom floats
\RequirePackage{flafter} % (builtin) floats after their definitions
\RequirePackage[table]{xcolor} % custom table cell colors
\RequirePackage{pgfplots} % includes tikz, xcolor, graphicx
\RequirePackage{adjustbox} % automatically scale to fit
\RequirePackage[scaled=0.95]{beramono} % custom teletype font
\RequirePackage{amsmath, amsfonts, amssymb} % improved math
\RequirePackage{bm} % bold math (must be after amsmath)
\RequirePackage{cancel} % strike-through in equations
\RequirePackage{listings} % code and algorithms
\RequirePackage{matlab-prettifier} % better MATLAB code formatting
\RequirePackage{upquote} % upright single quotes in verbatim
\RequirePackage{array} % custom column alignment types
\RequirePackage[absolute]{textpos} % free placement of objects
\RequirePackage{enumitem} % custom lists and list spacing
\RequirePackage{amsthm} % theorem environments
\RequirePackage{qrcode} % QR code generation
\RequirePackage{lastpage} % for page numbering (X of Y)
\RequirePackage{caption} % for \captionof command
\RequirePackage{hyperref} % hyperlinks

% TikZ libraries
\usetikzlibrary{math} % general math calculations in TikZ

% PGF Plot configuration
\pgfplotsset{compat=newest}

% ----------------------
% Added content features
% ----------------------

% Standard math macros
\ifdefineul
    \newcommand{\ul}[1] % vectors
        {{}\mkern1mu\underline{\mkern-1mu#1\mkern-1mu}\mkern1mu}
\fi
\DeclareSymbolFont{euler}{U}{eur}{m}{n}     % symbol font
\DeclareMathSymbol{\PI}{\mathord}{euler}{25} % up pi

% Convenience macros
\DeclareMathOperator{\sgn}{sgn}             % sign function
\DeclareMathOperator{\tr}{tr}               % trace of a matrix
\DeclareMathOperator{\adj}{adj}             % adjugate of a matrix
\DeclareMathOperator*{\argmax}{arg\,max}    % argmax operator
\DeclareMathOperator*{\argmin}{arg\,min}    % argmin operator
\newcommand{\abs}[1]{\left\lvert#1\right\rvert} % absolute value
\newcommand{\norm}[1]{\left\lVert#1\right\rVert} % vector or matrix norm
\newcommand{\ssm}[1]{[#1]_{\rule{0pt}{1.38ex}\times}} % skew symmetric matrix
\newcommand{\diff}[2]{\frac{\mathrm{d}{#1}}{\mathrm{d}{#2}}} % derivative
\newcommand{\piff}[2]{\frac{\partial{#1}}{\partial{#2}}} % partial derivative
\newcommand{\EE}[2]{\ensuremath{{{#1}\!\times\!10^{#2}}}} % scientific notation
\newcommand{\rom}[1]{\uppercase\expandafter{\romannumeral #1\relax}}

% Upright math letters
\newcommand{\upc}[1][]{\mathrm{c}_{#1}}     % abbreviated cosine
\newcommand{\ups}[1][]{\mathrm{s}_{#1}}     % abbreviated sine
\newcommand{\upt}[1][]{\mathrm{t}_{#1}}     % abbreviated tangent
\newcommand{\upd}{\mathrm{d}}               % differential operator
\newcommand{\upe}{\mathrm{e}}               % Euler's number
\newcommand{\upi}{\mathrm{i}}               % imaginary number
\newcommand{\upj}{\mathrm{j}}               % imaginary number
\newcommand{\upk}{\mathrm{k}}               % quaternion number

% Increase maximum number of matrix columns.
\setcounter{MaxMatrixCols}{20}

% Define custom table column types (array package) and spacings.
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}
\renewcommand{\arraystretch}{1.3} % spacing between matrix rows
\setlength{\tabcolsep}{1ex} % spacing between columns of a table

% Custom colors (xcolor package)
\definecolor{blue}{RGB}{76,76,255}      % 4c4cff
\definecolor{azure}{RGB}{0,128,255}     % 0080ff
\definecolor{cyan}{RGB}{0,209,209}      % 00d1d1
\definecolor{green}{RGB}{15,180,0}      % 0fb400
\definecolor{lime}{RGB}{180,216,0}      % b4d800
\definecolor{yellow}{RGB}{255,191,0}    % ffbf00
\definecolor{orange}{RGB}{255,128,0}    % ff8000
\definecolor{red}{RGB}{255,38,38}       % ff2626
\definecolor{magenta}{RGB}{255,0,255}   % ff00ff
\definecolor{purple}{RGB}{153,51,255}   % 9933ff

% Select the color theme.
\ifnum\@theme=0 % default, UCSD colors
    \definecolor{dark}{RGB}{24,43,73}    % UCSD Navy Blue
    \definecolor{light}{RGB}{198,146,20} % UCSD Gold
\fi
\ifnum\@theme=1 % Top Secret//SCI
    \definecolor{dark}{RGB}{55,31,13}
    \colorlet{light}{yellow}
\fi
\ifnum\@theme=2 % Top Secret
    \definecolor{dark}{RGB}{51,21,0}
    \colorlet{light}{orange}
\fi
\ifnum\@theme=3 % Secret
    \definecolor{dark}{RGB}{64,0,0}
    \colorlet{light}{red}
\fi
\ifnum\@theme=4 % Confidential
    \definecolor{dark}{RGB}{0,0,76}
    \colorlet{light}{blue}
\fi
\ifnum\@theme=5 % Controlled Unclassified
    \definecolor{dark}{RGB}{38,0,76}
    \colorlet{light}{purple}
\fi
\ifnum\@theme=6 % Uncontrolled Unclassified
    \definecolor{dark}{RGB}{7,38,4}
    \colorlet{light}{green}
\fi

% Clickable mailto email addresses
\newcommand{\mailto}[1]{\href{mailto:#1}{\nolinkurl{#1}}}

% Horizontal line with middle text
\newsavebox{\@btext}
\newlength{\@bwidth}
\newcommand{\breaktext}[2][light]{%
    \sbox{\@btext}{#2}%
    \setlength{\@bwidth}{0.5\textwidth}%
    \addtolength{\@bwidth}{-0.5\wd\@btext}%
    \addtolength{\@bwidth}{-1ex}%
    \par{\color{#1!20}\rule[0.5ex]{\@bwidth}{1pt}}%
    \hfill{\usebox{\@btext}}\hfill%
    {\color{#1!20}\rule[0.5ex]{\@bwidth}{1pt}}\par}

% Vertical and horizontal centering
\newenvironment{Center}
    {\topskip0pt\par\vspace*{\fill}\vspace*{-\parskip}\vspace*{-\topsep}%
    \begin{minipage}{\textwidth}\centering}
    {\end{minipage}\vspace*{\fill}\vspace*{-\topsep}\par}

% Arbitrary object placement (textpos package)
\newcommand{\pos}[4][]{%
    \if\relax\detokenize{#1}\relax
        \sbox{\@tempboxa}{#4}
        \begin{textblock*}{\wd\@tempboxa}(#2,#3)
            \usebox{\@tempboxa}
        \end{textblock*}
    \else
        \begin{textblock*}{#1}(#2,#3)
            #4
        \end{textblock*}
    \fi}

% Colored text box: \shadedbox[color]{text}
\definecolor{shade}{RGB}{206,224,242}
\newcommand{\shadedbox}[2][shade]{\setlength{\fboxsep}{5pt}%
    \hspace*{-5pt}%
    \colorbox{#1}{\parbox{\columnwidth}{#2}}% colorbox from xcolor package
    \hspace*{-5pt}}

% Themed callout boxes
\newcommand{\alertbox}[1]{\shadedbox[light!30]{%
    \textcolor{dark}{\textbf{Note:}} #1}}
\newcommand{\infobox}[1]{\shadedbox[dark!10]{%
    \textcolor{dark}{#1}}}
\newcommand{\highlightbox}[1]{\shadedbox[light!20]{%
    \textcolor{dark}{\textbf{#1}}}}

% QR code helper
\newcommand{\qrlink}[2][1.5cm]{\qrcode[height=#1]{#2}}

% Side-by-side figures: \twofigures{img1}{caption1}{img2}{caption2}
\newcommand{\twofigures}[4]{%
    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{#1}
        \captionof{figure}{#2}
    \end{minipage}\hfill
    \begin{minipage}{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{#3}
        \captionof{figure}{#4}
    \end{minipage}}

% Three figures: \threefigures{img1}{caption1}{img2}{caption2}{img3}{caption3}
\newcommand{\threefigures}[6]{%
    \begin{minipage}{0.31\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{#1}
        \captionof{figure}{#2}
    \end{minipage}\hfill
    \begin{minipage}{0.31\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{#3}
        \captionof{figure}{#4}
    \end{minipage}\hfill
    \begin{minipage}{0.31\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{#5}
        \captionof{figure}{#6}
    \end{minipage}}

% ========================================
% AI-Friendly Content Commands
% ========================================

% Structured content blocks
\newcommand{\slidegoal}[1]{\infobox{\textbf{Goal:} #1}\vspace{1ex}}
\newcommand{\context}[1]{\shadedbox[dark!5]{\textbf{Context:} #1}\vspace{1ex}}
\newcommand{\takeaway}[1]{\highlightbox{\textbf{Key Takeaway:} #1}}
\newcommand{\source}[1]{\vfill\hfill{\scriptsize\textit{Source: #1}}}

% Semantic slide types - Key Point
\newcommand{\keypoint}[2]{%
    \section{#1}
    \begin{Center}
        \Large\textcolor{dark}{#2}
    \end{Center}}

% Semantic slide types - Data Slide
\newcommand{\dataslide}[2]{%
    \section{#1}
    \alertbox{This slide contains data visualizations and quantitative information.}
    \vspace{2ex}
    #2}

% Semantic slide types - Timeline
\newcounter{timelineitem}
\newcommand{\timeline}[2]{%
    \section{#1}
    \setcounter{timelineitem}{0}
    \begin{itemize}
        #2
    \end{itemize}}
\newcommand{\timelineitem}[2]{%
    \stepcounter{timelineitem}
    \item[\textcolor{dark}{\textbf{\thetimelineitem.}}] \textbf{#1} --- #2}

% Auto-layout helpers - Split slide
\newcommand{\splitslide}[3]{%
    \section{#1}
    \begin{minipage}[t]{0.48\columnwidth}
        #2
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        #3
    \end{minipage}}

% Auto-layout helpers - Grid layout (2x2)
\newcommand{\gridlayout}[5]{%
    \section{#1}
    \begin{minipage}[t]{0.48\columnwidth}
        #2
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        #3
    \end{minipage}

    \vspace{2ex}

    \begin{minipage}[t]{0.48\columnwidth}
        #4
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        #5
    \end{minipage}}

% Auto-layout helpers - Icon item (using bullet as simple icon)
\newcommand{\iconitem}[2]{%
    \begin{minipage}{0.1\columnwidth}
        \centering\Large\textcolor{dark}{#1}
    \end{minipage}\hfill
    \begin{minipage}{0.85\columnwidth}
        #2
    \end{minipage}
    \vspace{1ex}}

% Content templates - Problem/Solution
\newcommand{\problemsolution}[3]{%
    \section{#1}
    \subsection{Problem}
    #2

    \subsection{Solution}
    #3}

% Content templates - Before/After
\newcommand{\beforeafter}[3]{%
    \section{#1}
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{Before}
        #2
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{After}
        #3
    \end{minipage}}

% Content templates - Pros/Cons
\newcommand{\proscons}[3]{%
    \section{#1}
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{Pros}
        \begin{itemize}
            #2
        \end{itemize}
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{Cons}
        \begin{itemize}
            #3
        \end{itemize}
    \end{minipage}}

% Content templates - Comparison
\newcommand{\comparison}[5]{%
    \section{#1}
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{#2}
        #3
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\columnwidth}
        \subsection{#4}
        #5
    \end{minipage}}

% Simple table helper: \simpletable{caption}{header row}{data rows}
\newcommand{\simpletable}[3]{%
    \begin{center}
        \begin{tabular}{#2}
            \hline
            #3
        \end{tabular}
        \captionof{table}{#1}
    \end{center}}

% Simple bar graph helper: \bargraph{caption}{labels}{values}
% Example: \bargraph{Sales Data}{Q1,Q2,Q3,Q4}{50,75,60,90}
\newcommand{\bargraph}[3]{%
    \begin{center}
        \begin{tikzpicture}
            \begin{axis}[
                ybar,
                bar width=0.6cm,
                xlabel={},
                ylabel={Value},
                symbolic x coords={#2},
                xtick=data,
                nodes near coords,
                nodes near coords align={vertical},
                ymin=0,
                width=0.9\columnwidth,
                height=5cm,
                xlabel style={font=\small},
                ylabel style={font=\small},
                tick label style={font=\footnotesize},
                every node near coord/.append style={font=\scriptsize},
                enlarge x limits=0.15
            ]
            \addplot[fill=light] coordinates {#3};
            \end{axis}
        \end{tikzpicture}
        \captionof{figure}{#1}
    \end{center}}

% ========================================
% Research Presentation Templates
% ========================================

% Research questions slide: \researchquestions{title}{questions}
% questions should be an itemize environment
\newcommand{\researchquestions}[2]{%
    \section{#1}
    \alertbox{\textbf{Research Questions:}}
    \vspace{1ex}
    #2}

% Results presentation: \resultslide{title}{figure}{caption}{commentary}
% Creates two-column layout with figure left, bullets right
\newcommand{\resultslide}[4]{%
    \section{#1}
    \begin{minipage}[t]{0.52\columnwidth}
        \begin{center}
            \includegraphics[width=\linewidth]{#2}
            \captionof{figure}{#3}
        \end{center}
    \end{minipage}\hfill
    \begin{minipage}[t]{0.45\columnwidth}
        \raggedright
        #4
    \end{minipage}}

% Methodology slide: \methodslide{title}{approach}{details}
\newcommand{\methodslide}[3]{%
    \section{#1}
    \infobox{\textbf{Approach:} #2}
    \vspace{2ex}
    #3}

% Multi-statement slide: \statements{title}{statement1}{statement2}{statement3}
\newcommand{\statements}[4]{%
    \section{#1}
    \begin{Center}
        \large #2

        \vspace{1.5ex}

        #3

        \vspace{1.5ex}

        #4
    \end{Center}}

% Experimental design matrix: \designmatrix{title}{tabular content}
\newcommand{\designmatrix}[2]{%
    \section{#1}
    \begin{center}
        #2
    \end{center}}

% ========================================
% End Research Presentation Templates
% ========================================

% ========================================
% End AI-Friendly Content Commands
% ========================================

% Quad charts: \quadchart{top-left}{top-right}{bottom-left}{bottom-right}
\newlength{\@qw}
\ifstandard
    \setlength{\@qw}{4.84cm}
\else
    \setlength{\@qw}{6.81cm}
\fi
\newcommand{\quadchart}[4]{%
    {\renewcommand{\arraystretch}{0}%
    \begin{tabular}{@{}c|c@{}}
        \begin{minipage}[t][3.22cm][t]{\@qw} % Top-left
            {#1}
        \end{minipage} &
        \begin{minipage}[t][3.22cm][t]{\@qw} % Top-right
            {#2}
        \end{minipage} \\
        \hline\\[1ex]
        \begin{minipage}[t][3.22cm][t]{\@qw} % Bottom-left
            {#3}
        \end{minipage} &
        \begin{minipage}[t][3.22cm][t]{\@qw} % Bottom-right
            {#4}
        \end{minipage}
    \end{tabular}}}

% -------------------------------------
% Custom code styles (listings package)
% -------------------------------------

% Pseudocode
\lstdefinestyle{pseudostyle}{
    xleftmargin=30pt, xrightmargin=0pt,
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=10pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\linespread{1.1}\ttfamily\small,
    commentstyle=\color{gray}\itshape,
    stringstyle=\color{orange},
    columns=flexible}
\lstdefinelanguage{pseudolang}{
    keywords={function, for, while, do, break, continue,
        if, then, else, end, in, to, until, return},
    keywordstyle=\color{black}\bfseries,
    morecomment=[l]{\%\ },
    morestring=[b]",
    morestring=[b]',
    mathescape=true,
    escapechar=`}
\lstnewenvironment{pseudocode}[1][1]{%
    \lstset{language=pseudolang, style=pseudostyle, firstnumber=#1}}{}
\lstnewenvironment{pseudocodetight}[1][1]{%
    \lstset{language=pseudolang, style=pseudostyle, firstnumber=#1,
        xleftmargin=15pt, numbersep=10pt, rulecolor=\color{white}}}{}

% Python code
\lstdefinestyle{pythonstyle}{
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{azure!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\ttfamily\footnotesize,
    commentstyle=\color{gray},
    stringstyle=\color{orange},
    keywordstyle=[1]\color{azure},   % keywords
    keywordstyle=[2]\color{azure},   % builtin functions
    keywordstyle=[3]\color{purple},  % NumPy
    keywordstyle=[4]\color{magenta}} % MatplotLib
\lstdefinelanguage{pythonlang}{
    keywords={and, as, assert, break, class, continue, def, del, elif, else,
        except, False, finally, for, from, global, if, import, in, is, lambda,
        None, nonlocal, not, or, pass, raise, return, True, try, while, with,
        yield},
    morekeywords=[2]{abs, aiter, all, any, anext, ascii, bin, bool, breakpoint,
        bytearray, bytes, callable, chr, classmethod, compile, complex, delattr,
        dict, dir, divmod, enumerate, eval, exec, exit, filter, float, format,
        frozenset, getattr, globals, hasattr, hash, help, hex, id, input, int,
        isinstance, issubclass, iter, len, list, locals, map, max, memoryview,
        min, next, object, oct, open, ord, quit, pow, print, property, range,
        repr, reversed, round, set, setattr, slice, sorted, staticmethod, str,
        sum, super, tuple, type, vars, zip},
    morekeywords=[3]{DataSource, LinAlgError, MachAr, NDArrayOperatorsMixin,
        NumpyVersion, abs, absolute, add, all, allclose, amax, amin, angle,
        any, append, apply_along_axis, apply_over_axes, arange, arccos, arccosh,
        arcsin, arcsinh, arctan, arctan2, arctanh, argmax, argmin, argpartition,
        argsort, argwhere, around, array, array2string, array_equal,
        array_equiv, array_repr, array_split, array_str, arraysetops,
        asanyarray, asarray, asarray_chkfinite, ascontiguousarray, asfarray,
        asfortranarray, asmatrix, asscalar, atleast_1d, atleast_2d, atleast_3d,
        average, bartlett, base_repr, binary_repr, bincount, bitwise_and,
        bitwise_or, bitwise_xor, blackman, block, bmat, broadcast,
        broadcast_arrays, broadcast_shapes, broadcast_to, busday_count,
        busday_offset, busdaycalendar, byte_bounds, c_, can_cast, capitalize,
        cbrt, ceil, center, chararray, chebyshev, cholesky, choose, cla, clip,
        column_stack, common_type, compare_chararrays, compress, concatenate,
        cond, conj, conjugate, convolve, copy, copysign, copyto, corrcoef,
        correlate, cos, cosh, count, count_nonzero, cov, cross, cumprod, cumsum,
        datetime_as_string, datetime_data, decode, deg2rad, degrees, delete,
        deprecate, deprecate_with_doc, det, diag, diag_indices,
        diag_indices_from, diagflat, diagonal, diff, digitize, disp, divide,
        divmod, dot, dsplit, dstack, dtype, ediff1d, eig, eigh, eigs, eigvals,
        eigvalsh, einsum, einsum_path, empty, empty_like, encode, endswith,
        equal, errstate, exp, exp2, expand_dims, expandtabs, expm1, extract,
        eye, fabs, fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn,
        ifftshift, ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn,
        fill_diagonal, find, find_common_type, finfo, fix, flat, flatiter,
        flatnonzero, flatten, flip, fliplr, flipud, float_power, floor,
        floor_divide, fmax, fmin, fmod, format_float_positional,
        format_float_scientific, format_parser, frexp, frombuffer, fromfile,
        fromfunction, fromiter, frompyfunc, fromregex, fromstring, fsolve, full,
        full_like, gcd, genfromtxt, geomspace, get_include, get_printoptions,
        getbufsize, geterr, geterrcall, geterrobj, gradient, greater,
        greater_equal, hamming, hanning, heaviside, hermite, hermite_e,
        histogram, histogram2d, histogram_bin_edges, histogramdd, hsplit,
        hstack, hypot, i0, identity, identiy, iinfo, imag, in1d, index, indices,
        inner, insert, interp, intersect1d, inv, invert, is_busday, isalnum,
        isalpha, isclose, iscomplex, iscomplexobj, isdecimal, isdigit, isfinite,
        isfortran, isin, isinf, islower, isnan, isnat, isneginf, isnumeric,
        isposinf, isreal, isrealobj, isscalar, issctype, isspace, issubclass_,
        issubdtype, issubsctype, istitle, isupper, ix_, join, kaiser, kron,
        laguerre, lcm, ldexp, left_shift, legendre, less, less_equal, lexsort,
        linspace, ljust, load, loadtxt, log, log10, log1p, log2, logaddexp,
        logaddexp2, logical_and, logical_not, logical_or, logical_xor, logspace,
        lower, lstrip, lstsq, mask_indices, mat, matmul, matrix_power,
        matrix_rank, max, maximum, maximum_sctype, may_share_memory, mean,
        median, memmap, meshgrid, mgrid, min, min_scalar_type, minimum,
        mintypecode, mod, modf, moveaxis, msort, multi_dot, multiply,
        nan_to_num, nanargmax, nanargmin, nancumprod, nancumsum, nanmax,
        nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd,
        nansum, nanvar, ndenumerate, ndindex, nditer, negative, nested_iters,
        nextafter, nonzero, norm, not_equal, obj2sctype, ode, ogrid, ones,
        ones_like, outer, packbits, pad, partition, pcolor, percentile, pi,
        piecewise, pinv, place, polyfit, polynomial, polyval, positive, power,
        printoptions, prod, promote_types, ptp, put, put_along_axis, putmask,
        qr, quad, quantile, r_, rad2deg, radians, rand, randn, ravel,
        ravel_multi_index, real, real_if_close, reciprocal, remainder, repeat,
        replace, repmat, require, reshape, resize, result_type, rfind,
        right_shift, rindex, rint, rjust, roll, rollaxis, rot90, round_, round,
        row_stack, rpartition, rsplit, rstrip, s_, save, savetxt, savez,
        savez_compressed, sctype2char, searchsorted, select, set_printoptions,
        set_string_function, setbufsize, setdiff1d, seterr, seterrcall,
        seterrobj, setxor1d, shape, shares_memory, show_config, sign, signbit,
        sin, sinc, sinh, slogdet, solve, sort, sort_complex, spacing, split,
        splitlines, sqrt, square, squeeze squeeze, stack, startswith, std,
        str_len, strip, subtract, sum, svd, swapaxes, swapcase, take,
        take_along_axis, tan, tanh, tensordot, tensorinv, tensorsolve, tile,
        title, tofile, tolist, trace, translate, transpose, trapz, tri, tril,
        tril_indices, tril_indices_from, trim_zeros, triu, triu_indices,
        triu_indices_from, true_divide, trunc, typename, union1d, unique,
        unpackbits, unravel_index, unwrap, upper, vander, var, vdot, vectorize,
        vsplit, vstack, where, who, xlabel, ylabel, zeros, zeros_like, zfill},
    morekeywords=[4]{acorr, angle_spectrum, annotate, arrow, autoscale, autumn,
        axes, axhline, axhspan, axis, axline, axvline, axvspan, bar, bar_label,
        barbs, barh, bone, box, boxplot, broken_barh, cla, clabel, clf, clim,
        close, cohere, colorbar, connect, contour, contourf, cool, copper, csd,
        delaxes, disconnect, draw, draw_if_interactive, errorbar, eventplot,
        figimage, figlegend, fignum_exists, figtext, figure, fill, fill_between,
        fill_betweenx, findobj, flag, gca, gcf, gci, get,
        get_current_fig_manager, get_figlabels, get_fignums, get_plot_commands,
        getp, ginput, gray, grid, hexbin, hist, hist2d, hlines, hot, hsv,
        imread, imsave, imshow, inferno, install_repl_displayhook, ioff, ion,
        isinteractive, get, legend, locator_params, loglog, magma,
        magnitude_spectrum, margins, matshow, minorticks_off, minorticks_on,
        new_figure_manager, nipy_spectral, pause, pcolor, pcolormesh,
        phase_spectrum, pie, pink, plasma, plot, plot_date, polar, prism, psd,
        quiver, quiverkey, rc, rc_context, rcdefaults, rgrids, savefig, sca,
        scatter, sci, semilogx, semilogy, set_cmap, setp, show, specgram,
        spring, spy, stackplot, stairs, stem, step, streamplot, subplot,
        subplot2grid, subplot_mosaic, subplot_tool, subplots, subplots_adjust,
        summer, suptitle, switch_backend, table, text, thetagrids, tick_params,
        ticklabel_format, tight_layout, title, tricontour, trcontourf,
        tripcolor, triplot, twinx, twiny, uninstall_repl_displayhook,
        violinplot, viridis, vlines, waitforbuttonpress, winter, xcorr, xkcd,
        xlabel, xlim, xscale, xticks, ylabel, ylim, yscale, yticks},
    morecomment=[l]{\#\ },
    morecomment=[s]{"""}{"""},
    morestring=[b]",
    morestring=[b]',
    prebreak={ \char92},
    escapechar=`}
\lstnewenvironment{python}[1][1]{%
    \lstset{language=pythonlang, style=pythonstyle, firstnumber=#1}}{}
\newcommand*{\inputpython}[3]{\lstinputlisting[linerange={{#2}-{#3}},
    firstnumber=#2, language=pythonlang, style=pythonstyle]{#1}}
\newcommand*{\py}{\lstinline[language=pythonlang, basicstyle=\ttfamily]}

% MATLAB code
\lstdefinestyle{matlabstyle}{
    style=Matlab-editor,
    basicstyle=\ttfamily\footnotesize,
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{green!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    prebreak={ ...},
    escapechar=`}
\lstnewenvironment{matlab}[1][1]{%
    \lstset{style=matlabstyle, firstnumber=#1}}{}
\newcommand*{\inputmatlab}[3]{\lstinputlisting[
    style=matlabstyle, linerange={{#2}-{#3}}, firstnumber=#2]{#1}}
\newcommand*{\mat}{\lstinline[style=Matlab-editor, basicstyle=\ttfamily]}

% R code
\lstdefinestyle{rstyle}{
    xleftmargin=40pt, xrightmargin=0pt,
    frame=leftline, framesep=9pt, framerule=2pt, rulecolor=\color{purple!20},
    numbers=left, numberstyle=\scriptsize\ttfamily\color{gray}, numbersep=20pt,
    upquote=true, showstringspaces=false, tabsize=4,
    breaklines=true, breakindent=20pt, breakatwhitespace=true,
    breakautoindent=true,
    aboveskip=0.8em, belowskip=0.2em,
    basicstyle=\linespread{1.2}\ttfamily\footnotesize,
    commentstyle=\color{green},
    stringstyle=\color{red},
    keywordstyle=\color{azure}}
\lstdefinelanguage{rlang}{
    language=R,
    morekeywords={TRUE, FALSE},
    deletekeywords={data, frame, length, as, character, gamma, beta},
    prebreak={},
    escapechar=`}
\lstnewenvironment{rlang}[1][1]{%
    \lstset{language=rlang, style=rstyle, firstnumber=#1}%
    \colorlet{ruleColor}{purple}}{}
\newcommand*{\inputrlang}[3]{\lstinputlisting[linerange={{#2}-{#3}},
    firstnumber=#2, language=rlang, style=rstyle]{#1}}
\newcommand*{\rlng}{\lstinline[language=rlang, basicstyle=\ttfamily]}

% Line highlighting
\newcommand{\HL}{\tikz[overlay]{\draw[black, line width=2pt]
    (-10pt,-4pt) -- (-10pt,10pt);}}

% ---------------------------
% General document formatting
% ---------------------------

% Default font family
\renewcommand{\sfdefault}{qhv} % TeX Gyre Heros (like Helvetica)
\renewcommand\familydefault{\sfdefault}

% Default plot settings
\pgfplotsset{%
    tick label style={font=\scriptsize\sffamily},
    every axis label={font=\sffamily},
    legend style={font=\tiny\sffamily},
    label style={font=\sffamily},
    nodes near coords style=
        {font=\scriptsize,/pgf/number format/assume math mode}
}

% Define margins.
\ifstandard
    \newlength{\@xw}\setlength{\@xw}{6cm}   % half page width
\else
    \newlength{\@xw}\setlength{\@xw}{8cm}   % half page width
\fi
\newlength{\@yh}\setlength{\@yh}{4.5cm}     % half page height
\newlength{\@mgnT}\setlength{\@mgnT}{1.5cm} % top margin
\newlength{\@mgnB}\setlength{\@mgnB}{0.999cm} % bottom margin
\newlength{\@mgnL}\setlength{\@mgnL}{1cm}   % left margin
\newlength{\@mgnC}\setlength{\@mgnC}{1cm}   % column separation
\newlength{\@mgnR}\setlength{\@mgnR}{1cm}   % right margin
\newlength{\@mgnS}\setlength{\@mgnS}{\@mgnT} % head separation
\addtolength{\@mgnS}{-1cm}
\ifstandard
    \addtolength{\@mgnS}{0.374cm}
\else
    \addtolength{\@mgnS}{0.348cm}
\fi

% Page geometry
\geometry{
    paperwidth=2\@xw, paperheight=2\@yh,
    top=\@mgnT, bottom=\@mgnB,
    left=\@mgnL, right=\@mgnR,
    columnsep=\@mgnC,
    headheight=1cm, headsep=\@mgnS}
\setlength{\columnseprule}{0.1pt}

% Derived dimensions
\newlength{\@xShift}\setlength{\@xShift}{\@xw} % full-page x shift
\addtolength{\@xShift}{-\@mgnL}
\newlength{\@yShift}\setlength{\@yShift}{\@mgnT} % full-page y shift
\ifstandard
    \addtolength{\@yShift}{-4.148cm}
\else
    \addtolength{\@yShift}{-4.078cm}
\fi
\newlength{\@xHead}\setlength{\@xHead}{-0.5\@mgnL} % header x position
\addtolength{\@xHead}{0.5\@mgnR}
\newlength{\@yHead}\setlength{\@yHead}{-4cm} % header y position

% Paragraph parameters
\setlength{\parindent}{0pt} % indentation of first line of paragraphs
\setlength{\parskip}{1.2ex plus 6pt} % space between paragraphs
\setlength{\emergencystretch}{\hsize} % overfull avoidance
\overfullrule=1mm % overfull black bands
\raggedbottom % page-ending empty space allowed
\DeclareRobustCommand\justifying{%
    \let\\\@normalcr
    \@rightskip\z@skip \rightskip\@rightskip
    \leftskip\z@skip
    \finalhyphendemerits=5000
    \parfillskip=0pt plus 1fil}

% Reduce list spacing (enumitem package).
\setlist{
    itemindent=0pt,
    labelsep=0.2cm,
    labelwidth=0.4cm,
    leftmargin=0.6cm,
    topsep=0pt,
    parsep=0.5\parskip,
    itemsep=0.5\parskip,
    before=\raggedright}

% Reduce spacing of other environments (enumitem package).
\setlist[trivlist]{topsep=0pt}

% Hyperlink styling
\hypersetup{
    colorlinks=true,
    linkcolor=light,
    citecolor=light,
    urlcolor=light,
    hypertexnames=false}
\def\UrlFont{\sffamily\itshape}


% Use French spacing.
\frenchspacing

% Theorem environments
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{note}{Note}

% Pretitle
\newif\ifpretitle\pretitlefalse
\def\@pretitle{}
\newcommand{\pretitle}[1]{\pretitletrue\def\@pretitle{#1}}
% Title
\newif\iftitle\titlefalse
\def\@title{}
\renewcommand{\title}[1]{\titletrue\def\@title{#1}}
% Subtitle
\newif\ifsubtitle\subtitlefalse
\def\@subtitle{}
\newcommand{\subtitle}[1]{\subtitletrue\def\@subtitle{#1}}
% Author
\newif\ifauthor\authorfalse
\def\@author{}
\renewcommand{\author}[1]{\authortrue\def\@author{#1}}
% Date
\newif\ifdate\datefalse
\def\@date{\today}
\renewcommand{\date}[1]{\datetrue\def\@date{#1}}
% Department
\newif\ifdepartment\departmentfalse
\def\@department{Department/School Name}
\newcommand{\department}[1]{\departmenttrue\def\@department{#1}}
% Affiliation
\newif\ifaffiliation\affiliationfalse
\def\@affiliation{}
\newcommand{\affiliation}[1]{\affiliationtrue\def\@affiliation{#1}}
% Presentation metadata (AI-friendly)
\newif\iftopic\topicfalse
\def\@topic{}
\newcommand{\topic}[1]{\topictrue\def\@topic{#1}}
\newif\ifaudience\audiencefalse
\def\@audience{}
\newcommand{\audience}[1]{\audiencetrue\def\@audience{#1}}
\newif\ifduration\durationfalse
\def\@duration{}
\newcommand{\duration}[1]{\durationtrue\def\@duration{#1}}
\newif\ifkeywords\keywordsfalse
\def\@keywords{}
\newcommand{\keywords}[1]{\keywordstrue\def\@keywords{#1}}

% Distribution and CUI (see https://www.dodcui.mil/Distribution-Statements/)
\newcommand{\DistA}{Distribution Statement A. Approved for public release:
    distribution is unlimited.}
\newcommand{\DistB}[3]{Distribution Statement B. Distribution authorized to U.S.
    Government agencies {#1} {#2}. Other requests for this document must be
    referred to {#3}.}
    % \DistB{category}{date of determination}{controlling DoD office}
\newcommand{\DistC}[3]{Distribution Statement C. Distribution authorized to U.S.
    Government agencies and their contractors {#1} {#2}. Other requests for this
    document must be referred to {#3}.}
    % \DistC{category}{date of determination}{controlling DoD office}
\newcommand{\DistD}[3]{Distribution Statement D. Distribution authorized to the
    Department of Defense and U.S. DoD contractors only {#1} {#2}. Other
    requests for this document must be referred to {#3}.}
    % \DistD{category}{date of determination}{controlling DoD office}
\newcommand{\DistE}[3]{Distribution Statement E. Distribution authorized to DoD
    Components only {#1} {#2}. Other requests for this document must be referred
    to {#3}.}
    % \DistE{category}{date of determination}{controlling DoD office}
\newcommand{\DistF}[2]{Distribution Statement F. Further distribution only as
    directed by {#1} {#2} or higher DoD authority.}
    % \DistE{controlling DoD office}{date of determination}
% Distribution A
\newif\ifDistA\DistAfalse
\def\distributionA{\DistAtrue}
% CUI (see DoDI 5200.48, p. 16)
\newif\ifControlled\Controlledfalse
\def\@cuiBlock{
    Controlled By: <Organization> \\
    CUI Category(ies): <category> \\
    Distribution: <DistB, ..., DistF command> \\
    POC: <name>, <phone>}
\newcommand{\cui}[1]{\Controlledtrue\def\@cuiBlock{#1}}
% Classification
\def\@classBlock{}
\newcommand{\classified}[1]{\Controlledtrue\def\@classBlock{#1}}
% Banner
\def\@banner{CUI}
\newcommand{\banner}[1]{\Controlledtrue\def\@banner{\MakeUppercase{#1}}}

% Contact info
\def\@contactinfo{%
    University of California San Diego \\
    \@department \\[1ex]
    9500 Gilman Drive \\
    La Jolla, CA 92093 \\[1ex]
    Email: \mailto{contact@ucsd.edu} \\
    \href{https://www.ucsd.edu}{\textit{www.ucsd.edu}}}
\newcommand{\contact}[1]{\def\@contactinfo{{#1}}}

% Footer text
\def\@footertext{}
\newcommand{\footertext}[1]{\def\@footertext{#1}}

% Copyright (e.g., \copyrighted[additional]{year}{author}). Note that the
% copyright symbol is permitted regardless of whether the work has been
% registered with the US Copyright Office, but it is also not required.
\def\@copyrightstatement{}
\newcommand{\copyrighted}[3][]{
    \def\@copyrightstatement{%
        \\[2ex] Copyright {#2} {#3} \\[1ex]
        All rights reserved. No part of this work may be reproduced or used in
        any manner without the prior written permission of the copyright owner,
        except for the use of brief quotations.
        \if\relax\detokenize{#1}\relax
        \else
            \\[1ex]
            {#1}
        \fi}}

% Disclaimer statement
\def\@disclaimerstatement{
    \\[2ex] The views expressed in this document are those of the author(s) and do
    not necessarily reflect the official policy or position of the University of
    California San Diego.}
\newcommand{\disclaimer}[1]{\def\@disclaimerstatement{\par{#1}}}

% Custom motto
\def\@ucsdmotto{} % Empty by default
\newcommand{\motto}[1]{\def\@ucsdmotto{\\[2ex] \itshape{#1}}}

% Custom logo
\newif\ifextralogo\extralogofalse
\newcommand{\extralogo}[1]{%
    \extralogotrue%
    \newsavebox\@extralogobox%
    \sbox{\@extralogobox}{\includegraphics[%
        width=1.732cm, height=1.5cm, keepaspectratio]{#1}}}

% UCSD Logo paths (relative to document root)
% Using transparent PNG seal for main logo, text logos for headers
\def\@ucsdseal{UCSD-Branding/ucsd-seal-transparent.png}
\def\@ucsdlogopath{UCSD-Branding/JPG-CMYK/UCSanDiegoLogo-BlueGold-CMYK.jpg}
\def\@ucsdlogoblue{UCSD-Branding/JPG-CMYK/UCSanDiegoLogo-Blue-CMYK.jpg}
\def\@ucsdlogowhite{UCSD-Branding/JPG-CMYK/UCSanDiegoLogo-BlueGold-CMYK.jpg}
\def\@ucsdlogoblack{UCSD-Branding/JPG-CMYK/UCSanDiegoLogo-Black-CMYK.jpg}
% Allow user to override logo paths
\newcommand{\ucsdlogo}[1]{\def\@ucsdlogopath{#1}}
\newcommand{\ucsdlogoblue}[1]{\def\@ucsdlogoblue{#1}}
\newcommand{\ucsdlogowhite}[1]{\def\@ucsdlogowhite{#1}}
\newcommand{\ucsdlogoblack}[1]{\def\@ucsdlogoblack{#1}}

% TikZ predefined drawings
\tikzset{
    titlepage/.pic={
        % Background
        \fill[color=light] ({3.4cm-\@xw},-1) --
            ({2.3cm-\@xw},4.5) -- (\@xw,4.5) -- (\@xw,-4.5) --
            ({2.7cm-\@xw},-4.5);
        \fill[color=dark] ({3.9cm-\@xw},-1) --
            ({2.8cm-\@xw},4.5) -- (\@xw,4.5) -- (\@xw,-4.5) --
            ({3.2cm-\@xw},-4.5);
        % UCSD Seal - left aligned, slightly below center
        \node[inner sep=0, anchor=west] at ({-\@xw+0.5cm},{-1cm})
            {\includegraphics[width=2.5cm]{\@ucsdseal}};
        % Extra logo if provided (below UCSD seal)
        \ifextralogo
            \node[inner sep=0, anchor=north] at ({-\@xw+1.75cm},{-2.5cm})
                {\usebox{\@extralogobox}};
        \fi
        % Text widths
        \newlength{\@titlelegalwidth}
        \setlength{\@titlelegalwidth}{2\@xw}
        \addtolength{\@titlelegalwidth}{-5.4cm}
        \newlength{\@titlewidth}
        \newlength{\@legalwidth}
        \ifControlled
            \setlength{\@titlewidth}{0.6\@titlelegalwidth}
            \addtolength{\@titlewidth}{-0.25cm}
            \setlength{\@legalwidth}{0.4\@titlelegalwidth}
            \addtolength{\@legalwidth}{-0.25cm}
        \else
            \setlength{\@titlewidth}{\@titlelegalwidth}
            \setlength{\@legalwidth}{0pt}
        \fi
        % Titles and authors
        \node[right, inner sep=0]
            at ({4.4cm-\@xw},0)
            {\parbox{\@titlewidth}{\raggedright%
            \ifpretitle
                \normalsize\color{white}{\@pretitle} \\[1.5ex]
            \fi
            \iftitle
                \Large\color{white}{\bfseries{\@title}} \\[1.5ex]
            \fi
            \ifsubtitle
                \large\color{light!75!white}{\bfseries{\@subtitle}}%
                \\[2ex]
            \fi
            \ifauthor
                \normalsize\color{white}{\@author}
            \fi
            \ifaffiliation
                \\[0.5ex]\small\color{white}{\textit{\@affiliation}}
            \fi
            \ifdate
                \\[1ex]\normalsize\color{white}{\@date}
            \fi}};
        % Classification
        \ifControlled
            % Classification block
            \node[above, inner sep=0, align=left,
                    text width=\@legalwidth, white]
                at ({\@xw-0.5cm-0.5\@legalwidth},{0.5cm-\@yh})
                {\begin{minipage}[t]{\@legalwidth}
                    \tiny\raggedright\setlength{\parindent}{-1em}%
                    \@cuiBlock \\[1em]
                    \@classBlock
                \end{minipage}};
            % Classification banners
            \node[inner sep=0, text depth=0, light]
                at (0,4) {\bfseries\@banner\strut};
            \node[inner sep=0, text depth=0, light]
                at (0,-4) {\bfseries\@banner\strut};
        \else
            \ifDistA
                % Public release
                \node[right, inner sep=0, white, font=\tiny]
                    at ({4.4cm-\@xw},{0.5cm-\@yh}) {\MakeUppercase\DistA};
            \fi
        \fi},
    fullColorBkgnd/.pic={
        \fill[light] (-\@xw,0) -- (-\@xw,4.5) --
            ({\@xw-0.9cm},4.5) -- (\@xw,0) -- (\@xw,-4.5) --
            ({-\@xw+0.9cm},-4.5);
        \fill[dark] (-\@xw,2.5) -- (-\@xw,4.5) --
            ({\@xw-1.4cm},4.5) -- (\@xw,-2.5) --
            (\@xw,-4.5) -- ({-\@xw+1.4cm},-4.5);},
    chapterpage/.pic={
        % Background
        \path (0,0) pic{fullColorBkgnd};
        % Text
        \node[inner sep=0, text width={1.53\@xw},
                white, align=flush center]
            at (0,0) {\huge\bfseries\strut\@headername\strut};
        % Classification
        \ifControlled
            \node[inner sep=0, text depth=0, light]
                at (0,4) {\bfseries\@banner\strut};
            \node[inner sep=0, text depth=0, light]
                at (0,-4) {\bfseries\@banner\strut};
        \fi},
    sectionpage/.pic={
        % Background
        \fill[color=light] ({4.0cm-\@xw},3.5) --
            ({3.8cm-\@xw},4.5) -- (\@xw,4.5) -- (\@xw,3.5);
        \fill[color=dark] ({4.5cm-\@xw},3.5) --
            ({4.3cm-\@xw},4.5) -- (\@xw,4.5) -- (\@xw,3.5);
        % UCSD Logo in header
        \node[inner sep=0, anchor=west] at ({-\@xw+0.5cm},{4.03cm})
            {\includegraphics[height=0.6cm]{\@ucsdlogowhite}};
        % Title
        \node[right, inner sep=0, white, text depth=0]
            at ({-\@xw+4.6cm},4.03)
            {\begin{adjustbox}{max width=7cm}
                    \large\bfseries\ignorespaces\@headername\strut
                \end{adjustbox}};
        \phantomsection\addcontentsline{toc}{section}{\@altheadername}
        % Page number and banner
        \ifControlled
            \node[left, inner sep=0, text depth=0, light]
                at ({\@xw-0.5cm},4)
                {\bfseries\@banner~\thepage\ of \pageref{LastPage}\strut};
            \node[left, inner sep=0, text depth=0, dark]
                at ({\@xw-0.5cm},-4)
                {\bfseries\@banner\strut};
        \else
            \node[left, inner sep=0, text depth=0, light]
                at ({\@xw-0.5cm},4) {\bfseries\thepage\ of \pageref{LastPage}\strut};
        \fi
        % Footer text (if provided)
        \if\relax\detokenize{\@footertext}\relax\else
            \node[inner sep=0, text depth=0, white, font=\tiny]
                at (0,-4) {\@footertext\strut};
        \fi},
    closingpage/.pic={
        % Background
        \path (0,0) pic{fullColorBkgnd};
        % UCSD Seal
        \node[inner sep=0] at (-0.5\@xw,0)
            {\includegraphics[width=3.5cm]{\@ucsdseal}};
        % Text
        \hypersetup{urlcolor=white}
        \node[right, inner sep=0, white] at (0,0)
            {\parbox{0.8\@xw}{%
                \scriptsize\raggedright%
                \@contactinfo
                \@copyrightstatement
                \@disclaimerstatement
                \@ucsdmotto}};
        \hypersetup{urlcolor=light}
        % Classification
        \ifControlled
            \node[inner sep=0, text depth=0, light]
                at (0,4) {\bfseries\@banner\strut};
            \node[inner sep=0, text depth=0, light]
                at (0,-4) {\bfseries\@banner\strut};
        \fi}}

% Headers and footers
\newcommand{\@headername}{}
\pagestyle{fancy}
\fancyhead[L,R]{}
\fancyhead[C]{\tikz[overlay]{\path (\@xHead,\@yHead) pic{sectionpage};}}
\fancyfoot[L,C,R]{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Title page
\renewcommand{\maketitle}{
    \clearpage\thispagestyle{empty}\justifying\onecolumn%
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{titlepage};}
    \clearpage}

% Chapter page
\newcommand{\chapter}[2][]{%
    \clearpage\onecolumn\justifying\thispagestyle{empty}\normalsize%
    \renewcommand{\@headername}{#2}%
    \ifx&#1& % if parameter 1 is empty
        \def\@altheadername{#2}
    \else % if parameter 1 is not empty
        \def\@altheadername{#1}
    \fi
    \phantomsection\addcontentsline{toc}{chapter}{\@altheadername}
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{chapterpage};}%
    \clearpage}

% Header name and sections
\renewcommand{\section}[2][]{%
    \clearpage\justifying\onecolumn\normalsize%
    \renewcommand{\@headername}{#2}%
    \ifx&#1& % if parameter 1 is empty
        \def\@altheadername{#2}
    \else % if parameter 1 is not empty
        \def\@altheadername{#1}
    \fi}

% Other sections
\renewcommand{\subsection}[1]{\par\textcolor{light}{\textbf{#1}}\par}
\renewcommand{\subsubsection}[1]{\par\textbf{#1}\par}
\renewcommand{\paragraph}[1]{\par\textit{#1}\par}
\renewcommand{\subparagraph}[1]{\par\textit{#1}\par}

% Blank page for arbitrarily-placed graphics and text.
\newcommand\blankpage{\clearpage\null\thispagestyle{empty}}

% Section outline slide
% Usage: \outline{Section 1}{Section 2}{Section 3}{...}
\newcommand{\outline}[1]{
    \clearpage\justifying\onecolumn\normalsize%
    \renewcommand{\@headername}{Outline}%
    \def\@altheadername{Outline}%
    \begin{Center}
        \Large\textcolor{dark}{\textbf{Outline}}
        \vspace{2ex}

        \normalsize
        \begin{itemize}
            #1
        \end{itemize}
    \end{Center}
    \clearpage}

% Closing page
\newcommand{\closing}{
    \clearpage\thispagestyle{empty}\justifying\onecolumn\normalsize%
    \tikz[overlay, xshift=\@xShift, yshift=\@yShift]{
        \path (0,0) pic{closingpage};}%
    \clearpage}
