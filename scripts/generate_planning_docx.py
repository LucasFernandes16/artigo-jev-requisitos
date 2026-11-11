#!/usr/bin/env python3
"""Gera o documento executivo/acadêmico de planejamento em formato .docx para alinhamento da equipe."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figures"
OUT_DOCX = REPO_ROOT / "Planejamento_Pesquisa_JEV_MASTeR.docx"

NAVY = RGBColor(0x1B, 0x36, 0x5D)
TEAL = RGBColor(0x0D, 0x6E, 0x6E)
DARK_TEXT = RGBColor(0x22, 0x22, 0x22)
MUTED_TEXT = RGBColor(0x55, 0x55, 0x55)


def generate_figures() -> tuple[Path, Path]:
    """Gera gráficos de alta resolução (300 DPI) para ilustrar o diagnóstico empírico e a arquitetura."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figura 1: Diagnóstico empírico de Roßkothen et al. (Over-editing + Queda por Split Factor)
    diag_path = FIG_DIR / "baseline_diagnostic.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=300)

    metrics = ["CorpBERT F1", "EMD", "MASTeR F1", "Cond. F1", "Composto"]
    one_shot_vals = [0.834, 0.770, 0.561, 0.758, 0.811]
    llm_judge_vals = [0.823, 0.762, 0.553, 0.742, 0.797]

    x = range(len(metrics))
    w = 0.35
    bars1 = ax1.bar([i - w / 2 for i in x], one_shot_vals, width=w, label="Qwen3 Sem Juiz (One-Shot)", color="#2b5c8f")
    bars2 = ax1.bar([i + w / 2 for i in x], llm_judge_vals, width=w, label="Judge-Qwen3 (Loop LLM-Judge)", color="#d95f02")
    ax1.set_ylim(0.45, 0.96)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics, fontsize=8.5, rotation=12)
    ax1.set_ylabel("Pontuação Normalizada [0 - 1]", fontsize=9)
    ax1.set_title("A) Paradoxo de Over-Editing no Loop Generativo", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in list(bars1) + list(bars2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}", ha="center", fontsize=7)

    sf_labels = ["SF = 1\n(n=168)", "SF = 2\n(n=43)", "SF >= 3\n(n=38)"]
    sf_bert = [0.864, 0.766, 0.708]
    sf_master = [0.601, 0.478, 0.427]
    ax2.plot(sf_labels, sf_bert, marker="o", linewidth=2.2, color="#1b7837", label="CorpBERT F1 (Semântico)")
    ax2.plot(sf_labels, sf_master, marker="s", linewidth=2.2, color="#b2182b", label="MASTeR Main F1 (Sintático)")
    for i, (vb, vm) in enumerate(zip(sf_bert, sf_master, strict=True)):
        ax2.annotate(f"{vb:.3f}", (i, vb + 0.018), ha="center", fontsize=8, color="#1b7837", fontweight="bold")
        ax2.annotate(f"{vm:.3f}", (i, vm - 0.038), ha="center", fontsize=8, color="#b2182b", fontweight="bold")
    ax2.set_ylim(0.35, 0.95)
    ax2.set_title("B) Queda de Qualidade por Complexidade (Split Factor)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(diag_path, bbox_inches="tight")
    plt.close(fig)

    # Figura 2: Diagrama visual limpo da Arquitetura JEV-Cascade (1 + 2)
    arch_path = FIG_DIR / "jev_cascade_diagram.png"
    fig2, ax = plt.subplots(figsize=(10.6, 4.8), dpi=300)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(-0.35, 5.2)
    ax.axis("off")

    def draw_box(x0: float, y0: float, w0: float, h0: float, title: str, subtitle: str, fc: str, ec: str) -> None:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w0, h0, boxstyle="round,pad=0.10", facecolor=fc, edgecolor=ec, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(x0 + w0 / 2, y0 + h0 * 0.68, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1a1a1a")
        ax.text(x0 + w0 / 2, y0 + h0 * 0.30, subtitle, ha="center", va="center", fontsize=7.8, color="#333333")

    draw_box(0.3, 3.85, 2.7, 1.0, "1. Requisito Fonte (S)", "Texto livre em LN\n(249 reqs / 5 projetos)", "#eef4fa", "#2b5c8f")
    draw_box(3.7, 3.85, 3.8, 1.0, "2. LLM Gerador (JSON)", "Gemini 2.5 Flash / Qwen3-235B\nFormalize & Split -> {R_1..R_k}", "#eef4fa", "#2b5c8f")
    draw_box(
        1.4, 2.05, 6.7, 1.18,
        "3. JEV System One (1 Passe Paralelo por R_i | ~100-300 ms)",
        "Estado (S, R_i) -> 2x Choice (Template/Cond) + 6x Noul Acionáveis + 1x Score (C4..C1)\n[Zero tokens gerados | Elimina Feedback Filter por construção]",
        "#e6f5f3", "#0d6e6e",
    )

    draw_box(0.3, 0.25, 2.9, 1.05, "PATH A: Fast Accept", "Todos p_j >= 0.75 & Score >= C2\nAceita direto (0 custo extra)", "#e8f6e9", "#2e7d32")
    draw_box(3.6, 0.25, 3.0, 1.05, "PATH B: Guided Rewrite", "Violação clara (p_j < 0.40)\nMapa determinístico -> Gerador", "#fff8e1", "#ef6c00")
    draw_box(7.0, 0.25, 2.9, 1.05, "PATH C: Escalate Humano", "Zona incerta (0.40 <= p_j < 0.75)\nou falha após M ciclos", "#fdecea", "#c62828")

    arrow_kw = dict(arrowstyle="-|>,head_width=0.25,head_length=0.45", lw=1.6, color="#444444")
    ax.annotate("", xy=(3.6, 4.35), xytext=(3.1, 4.35), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.1, 3.33), xytext=(5.1, 3.78), arrowprops=arrow_kw)
    ax.annotate("", xy=(1.75, 1.40), xytext=(3.3, 1.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.1, 1.40), xytext=(5.1, 1.98), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.45, 1.40), xytext=(6.7, 1.98), arrowprops=arrow_kw)

    # Seta externa de retorno do Path B contornando pela direita sem cruzar caixas
    ax.plot([5.1, 5.1, 10.35, 10.35], [0.15, -0.18, -0.18, 4.35], ls="--", lw=1.6, color="#ef6c00")
    ax.annotate(
        "",
        xy=(7.6, 4.35),
        xytext=(10.35, 4.35),
        arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.45", lw=1.6, ls="--", color="#ef6c00"),
    )
    ax.text(9.45, 3.65, "Feedback determinístico\n(até M iterações)", fontsize=7.8, color="#b25000", ha="center", fontweight="bold")

    fig2.savefig(arch_path, bbox_inches="tight")
    plt.close(fig2)
    return diag_path, arch_path


def set_cell_shading(cell, color_hex: str) -> None:
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout_box(doc: Document, title: str, body: str, fill_hex: str = "EEF4FA", border_hex: str = "1B365D") -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, fill_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f"</w:tcBorders>"
    )
    tcPr.append(tcBorders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(title + "\n")
    r_title.bold = True
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = NAVY
    r_body = p.add_run(body)
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = DARK_TEXT
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_styled_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "Table Grid"

    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        set_cell_shading(cell, "1B365D")
        set_cell_margins(cell, top=110, bottom=110, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(rows, start=1):
        bg = "F7F9FC" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 and len(val) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            run.font.size = Pt(9.2)
            run.font.color.rgb = DARK_TEXT
            if j == 0:
                run.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_heading_1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = NAVY


def add_heading_2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11.5)
    run.font.color.rgb = TEAL


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(3)
    r1 = p.add_run(bold_prefix)
    r1.bold = True
    r1.font.size = Pt(10.2)
    r1.font.color.rgb = DARK_TEXT
    r2 = p.add_run(text)
    r2.font.size = Pt(10.2)
    r2.font.color.rgb = DARK_TEXT


def build_docx() -> Path:
    diag_fig, arch_fig = generate_figures()

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.3)
        section.right_margin = Cm(2.3)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = DARK_TEXT

    # Cabeçalho / Capa Executiva Compacta
    p_tag = doc.add_paragraph()
    p_tag.paragraph_format.space_after = Pt(2)
    r_tag = p_tag.add_run("DOCUMENTO DE ALINHAMENTO DE PESQUISA  |  ANÁLISE AVANÇADA DE REQUISITOS")
    r_tag.bold = True
    r_tag.font.size = Pt(9)
    r_tag.font.color.rgb = TEAL

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run(
        "Validação Discriminativa Calibrada (JEV-Cascade) na Conversão de Requisitos em Linguagem Natural para Templates MASTeR"
    )
    r_title.bold = True
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Estudo derivado de Roßkothen et al. (34th IEEE International Requirements Engineering Conference - RE 2026)  •  Formato Alvo: IEEEtran (Português)"
    )
    r_sub.italic = True
    r_sub.font.size = Pt(10)
    r_sub.font.color.rgb = MUTED_TEXT

    add_callout_box(
        doc,
        "Objetivo deste Documento para a Dupla",
        "Apresentar de forma clara o problema científico identificado no artigo base, a arquitetura proposta combinando LLMs geradores com o modelo de decisão rápida JEV (System One), a matriz de experimentos nos 249 requisitos oficiais e os pontos abertos para revisão e divisão de tarefas.",
    )

    # Seção 1
    add_heading_1(doc, "1. Contexto: O Artigo Base (Roßkothen et al., IEEE RE 2026)")
    p = doc.add_paragraph(
        "Em projetos industriais (especialmente automotivos, espaciais e ferroviários), especificações em linguagem natural (LN) livre sofrem de ambiguidade e falta de atomicidade. Templates semiformais como o MASTeR (SOPHIST) impõem regras gramaticais rígidas — como verbos modais obrigatórios (shall, should, will), identificação explícita do sistema/ator e prefixos condicionais padronizados (If, As soon as, As long as) — elevando drasticamente a qualidade e testabilidade dos requisitos."
    )
    p.paragraph_format.space_after = Pt(6)

    p2 = doc.add_paragraph(
        "No artigo base ('On Converting Natural Language Requirements into Semi-Formal Templates Using LLMs'), os autores em parceria com a Vector Informatik propuseram automatizar essa migração usando um Reflection Loop (ciclo de reflexão): um LLM gera e divide 1 requisito livre em k requisitos MASTeR atômicos e, em seguida, validadores baseados em LLM-as-a-Judge avaliam sintaxe e critérios do guia INCOSE, pedindo reescritas iterativas. Os autores publicaram no Zenodo (DOI: 10.5281/zenodo.20698022) o dataset completo com 249 requisitos fonte, 410 requisitos MASTeR escritos por humanos, prompts Jinja2 e resultados."
    )
    p2.paragraph_format.space_after = Pt(6)

    # Seção 2
    add_heading_1(doc, "2. Diagnóstico Crítico: Os 3 Gargalos que Encontramos no Código e nos Dados Deles")
    doc.add_paragraph(
        "Ao inspecionar o código oficial (simple_loop_implementation.py) e reprocessar os 996 registros do pacote de replicação dos autores, descobrimos três problemas concretos que justificam fortemente nosso artigo:"
    )

    add_bullet(
        doc,
        "Gargalo 1 — Explosão Multiplicativa de Chamadas e Custo (2k + 1 chamadas por ciclo): ",
        "Para cada requisito dividido em k sentenças MASTeR (média de 1,65, chegando a k=11), CADA iteração do loop original executa k chamadas ao Template Validator (reenviando 1.400 tokens de regras toda vez) + k chamadas ao INCOSE Validator (gerando texto para 9 categorias) + 1 chamada extra a um LLM de Feedback Filter. Esse filtro extra só existe porque o INCOSE Validator deles não enxerga o requisito original e vive cobrando números inexistentes (induzindo alucinação). Um único requisito com k=3 em 2 iterações chega a 17 chamadas completas de LLM!",
    )
    add_bullet(
        doc,
        "Gargalo 2 — Paradoxo de Over-Editing sem Calibração: ",
        "Nos dados oficiais dos próprios autores, o modelo Qwen3-235B SEM validadores (minimal-qwen235) obteve 68% de aprovação humana (C1/C2) e notas automáticas ligeiramente MAIORES que o loop completo com juízes (full-qwen235: CorpBERT F1 caiu de 0,834 para 0,823; score composto caiu de 0,811 para 0,797). Ou seja: passar 100% dos requisitos por juízes generativos não calibrados gasta muito mais e ainda estraga rascunhos simples que já estavam ótimos.",
    )
    add_bullet(
        doc,
        "Gargalo 3 — Colapso Silencioso em Requisitos Complexos (Sem Escalonamento): ",
        "Quando o requisito exige 3 ou mais divisões (Split Factor >= 3), a precisão de template MASTeR despenca de 0,601 para 0,427 (-28,9%). Os autores concluem que casos complexos exigem revisão humana, mas o sistema deles não tem probabilidades calibradas para saber QUANDO escalar.",
    )

    add_styled_table(
        doc,
        ["Configuração Oficial (Zenodo)", "CorpBERT F1", "EMD", "MASTeR F1", "Cond. F1", "BLEU", "Score Composto"],
        [
            ["minimal-qwen235 (Sem Juiz)", "0,834", "0,770", "0,561", "0,758", "0,460", "0,811"],
            ["full-qwen235 (Loop LLM-Judge)", "0,823", "0,762", "0,553", "0,742", "0,423", "0,797"],
            ["minimal-qwen3-next (Thinking)", "0,822", "0,758", "0,592", "0,725", "0,412", "0,801"],
            ["full-gpt-5 (Loop GPT-5)", "0,787", "0,725", "0,569", "0,656", "0,273", "0,753"],
            ["full-qwen235 [Estrato SF = 1]", "0,864", "0,785", "0,601", "-", "-", "-"],
            ["full-qwen235 [Estrato SF >= 3]", "0,708", "0,691", "0,427", "-", "-", "-"],
        ],
    )

    p_img1 = doc.add_paragraph()
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img1.add_run().add_picture(str(diag_fig), width=Inches(6.2))
    p_cap1 = doc.add_paragraph("Figura 1: Diagnóstico empírico extraído dos dados oficiais de Roßkothen et al. (2026).")
    p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap1.runs[0].italic = True
    p_cap1.runs[0].font.size = Pt(9)
    p_cap1.runs[0].font.color.rgb = MUTED_TEXT

    # Seção 3
    add_heading_1(doc, "3. Nossa Proposta: Arquitetura JEV-Cascade Reflection Loop")
    doc.add_paragraph(
        "O JEV (TypeSafe AI) é um modelo discriminativo 'System One': ao contrário de um LLM tradicional que gera texto token por token para justificar uma nota, o JEV recebe um objeto de estado estruturado e responde múltiplas perguntas tipadas simultaneamente num único passe paralelo (~100 a 300 ms), devolvendo probabilidades calibradas [0..1] e métricas de confiança."
    )

    add_heading_2(doc, "3.1 Como combinamos Validação no Loop + Cascata Calibrada")
    add_bullet(
        doc,
        "1 Passe Paralelo Condicionado à Fonte (S, R_i): ",
        "O LLM gerador continua escrevendo os candidatos MASTeR. Para cada candidato R_i, enviamos (S, R_i) ao JEV avaliando 9 primitivas de uma só vez: 2x Choice (qual Template Principal e qual Condição foram usados), 6x Noul (checklists binários calibrados: verbo modal, sintaxe MASTeR, atomicidade/singularidade, ator explícito, preservação de detalhes de S e ausência de valores alucinados) e 1x Score (rubrica C4..C1).",
    )
    add_bullet(
        doc,
        "Eliminação por Construção do Feedback Filter: ",
        "Como todas as 6 verificações Noul já olham para o requisito fonte S e só cobrem erros gramaticais ou omissões/alucinações em relação a S, o JEV nunca pede dados inventados. Cada item reprovado mapeia deterministicamente para uma frase curta de correção — cortando 100% das chamadas ao LLM de Feedback Filter!",
    )
    add_bullet(
        doc,
        "Roteamento Triplo Calibrado (Accept / Rewrite / Escalate): ",
        "(Path A - Fast Accept) Se todas as probabilidades p_j >= 0,75 e qualidade >= C2, aceita na hora sem gastar novos tokens e sem risco de over-editing; (Path B - Guided Rewrite) Se há violação clara (p_j < 0,40), envia apenas as instruções determinísticas das regras violadas para reescrita (até M=3 ciclos); (Path C - Escalate) Se cai na zona de incerteza (0,40 <= p_j < 0,75) ou esgota M ciclos, encaminha para revisão humana.",
    )

    p_img2 = doc.add_paragraph()
    p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img2.add_run().add_picture(str(arch_fig), width=Inches(6.2))
    p_cap2 = doc.add_paragraph("Figura 2: Fluxo do JEV-Cascade Reflection Loop implementado no repositório.")
    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap2.runs[0].italic = True
    p_cap2.runs[0].font.size = Pt(9)
    p_cap2.runs[0].font.color.rgb = MUTED_TEXT

    # Seção 4
    add_heading_1(doc, "4. Questões de Pesquisa (RQs) e Matriz Experimental")
    add_bullet(
        doc,
        "RQ1 (Eficácia de Formalização): ",
        "O loop guiado por JEV mantém ou supera a qualidade sintática (MASTeR F1) e semântica (BLEU / BERTScore / EMD) frente ao One-Shot e ao loop tradicional com LLM-as-a-Judge?",
    )
    add_bullet(
        doc,
        "RQ2 (Eficiência Operacional e Economia): ",
        "Qual a redução percentual em latência de validação (ms), consumo de tokens e custo financeiro por 1.000 requisitos proporcionada pelo JEV-Cascade?",
    )
    add_bullet(
        doc,
        "RQ3 (Calibração do Escalonamento por Complexidade): ",
        "A incerteza calibrada do JEV consegue concentrar o escalonamento humano nos requisitos mais complexos (Split Factor >= 2), elevando a qualidade garantida do lote aceito automaticamente?",
    )

    add_styled_table(
        doc,
        ["Bloco", "Configuração", "LLM Gerador", "Validador no Loop", "Objetivo Científico"],
        [
            ["Ref.", "Baselines Zenodo", "Qwen3-235B / GPT-5", "Nenhum / GPT-OSS-120B", "Comparação histórica direta com IEEE RE 2026"],
            ["Bloco A", "gemini_oneshot", "Gemini 2.5 Flash", "Nenhum (0 ciclos)", "Baseline inferior de custo/latência no Gemini"],
            ["Bloco A", "gemini_llm_judge", "Gemini 2.5 Flash", "Loop LLM-Judge + Filter", "Custo e eficácia da arquitetura original controlada"],
            ["Bloco A", "gemini_jev_cascade", "Gemini 2.5 Flash", "JEV System One + Cascata", "Avaliar ganho isolado do JEV (Validade Interna)"],
            ["Bloco B", "qwen235_jev_cascade", "Qwen3-235B (OpenRouter)", "JEV System One + Cascata", "Comparação 1:1 com o gerador principal do artigo"],
        ],
    )

    # Seção 5
    add_heading_1(doc, "5. Status Atual: O que já está pronto no Repositório")
    add_bullet(doc, "Dataset e Prompts Oficiais: ", "Baixados e validados (249 requisitos fonte / 410 MASTeR humanos) com os mesmos templates Jinja2 dos autores.")
    add_bullet(doc, "Código Python Modular (src/req_jev/): ", "Integração assíncrona pronta com Google Gemini, OpenRouter (Qwen3-235B) e API Oficial TypeSafe AI (typesafe-sdk), com checkpoint automático em JSONL (pode pausar e retomar sem perder dados) e suíte de testes passando.")
    add_bullet(doc, "Artigo IEEE em Português (paper/): ", "Estrutura completa em IEEEtran pronta para subir no Overleaf (Introdução, Fundamentação, Abordagem com diagrama TikZ, Metodologia, Tabela I real de baseline, Ameaças à Validade e Conclusão).")

    # Seção 6 - Perguntas para o Parceiro de Equipe
    add_heading_1(doc, "6. Pontos para Revisão e Opinião do Parceiro de Equipe")
    add_callout_box(
        doc,
        "Roteiro Rápido para Alinhamento da Dupla",
        "1) Escopo de Modelos: Concorda em rodarmos o estudo completo no Gemini 2.5 Flash (3 modos controlados) + 1 rodada com Qwen3-235B via OpenRouter (~US$ 1-2 de custo total) para comparação direta com o artigo?\n"
        "2) Checklists do JEV: As 6 regras Noul escolhidas (verbo modal, sintaxe MASTeR, singularidade, ator explícito, completude em relação à fonte e zero alucinação numérica) cobrem bem os erros práticos ou sugere adicionar/ajustar alguma?\n"
        "3) Limiares da Cascata: Vamos testar num piloto de 10 requisitos os cortes tau_reject = 0,40 e tau_accept = 0,75. Queremos priorizar maior automação (aceitar mais) ou maior rigor (escalar mais para humano)?\n"
        "4) Divisão Sugerida de Trabalho:\n"
        "   • Frente A (Experimentos & Análise Quantitativa): Rodar scripts nos 249 requisitos, gerar gráficos/tabelas finais e testes estatísticos de Wilcoxon.\n"
        "   • Frente B (Inspeção Qualitativa & Escrita no Overleaf): Analisar exemplos reais onde o JEV corrigiu erros ou escalou requisitos difíceis (estudo de caso qualitativo valoriza muito na nota!) e refinar o texto final.",
        fill_hex="FFF8E7",
        border_hex="D95F02",
    )

    doc.save(str(OUT_DOCX))
    return OUT_DOCX


if __name__ == "__main__":
    out = build_docx()
    print(f"Documento gerado com sucesso em: {out}")
