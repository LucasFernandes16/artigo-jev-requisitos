# Como Compilar no Overleaf

Esta pasta (`paper/`) foi estruturada de forma autocontida no template oficial **`IEEEtran` (`conference`)** em Português, incluindo diagramas vetoriais nativos em **TikZ** (não depende de imagens externas para compilar).

## Passo a Passo para Subir no Overleaf

1. Compacte o conteúdo da pasta `paper/`:
   ```bash
   cd /home/lucasfbm/Faculdade/artigo-requisitos-jev/paper
   zip -r ../artigo_overleaf.zip main.tex references.bib sections/ tables/
   ```
2. No **Overleaf**, clique em **New Project $\rightarrow$ Upload Project** e envie o arquivo `artigo_overleaf.zip`.
3. Certifique-se nas configurações do projeto Overleaf (*Menu*) de que:
   - **Compiler:** `pdfLaTeX`
   - **Main document:** `main.tex`
4. Após rodar seus experimentos completos no repositório e executar:
   ```bash
   .venv/bin/python scripts/evaluate_and_export_latex.py
   ```
   basta atualizar o arquivo `tables/experimental_results_auto.tex` (e os valores comentados com `% TODO` em `sections/05_resultados_discussao.tex`) no Overleaf e recompilar.
