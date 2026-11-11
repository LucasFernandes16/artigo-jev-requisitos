# JEV-Cascade Reflection Loop para Formalização de Requisitos em Templates MASTeR

Repositório experimental e código-fonte do artigo científico (formato **IEEEtran**) desenvolvido para a disciplina de **Análise Avançada de Requisitos**, estendendo o trabalho de Roßkothen et al. (*"On Converting Natural Language Requirements into Semi-Formal Templates Using LLMs"*, 34th IEEE RE 2026).

---

## Visão Geral da Proposta

No artigo base ([Rosskothen_RE2026 (2).pdf](Rosskothen_RE2026%20(2).pdf)), cada iteração do *Reflection Loop* executa $2k + 1$ chamadas autorregressivas de LLM para validar $k$ requisitos candidatos (*Template Validator*, *INCOSE Validator* de 9 critérios e *Feedback Filter*). Além do custo e latência elevados em lote, a análise empírica dos dados oficiais dos autores revela que o loop generativo sem calibração sofre de *over-editing* em requisitos simples (`CorpBERTF1` cai de `0.834` no `minimal-qwen235` para `0.823` no `full-qwen235`).

Este repositório implementa o **JEV-Cascade Reflection Loop**, substituindo a pilha de validadores generativos e o filtro de feedback por uma única inferência paralela **System One** via **JEV** (`typesafe-sdk`):

1. **Validação Tipada Condicionada à Fonte (`Choice` + `Noul` + `Score`):** Avalia $(S, R_i)$ em milissegundos sem gerar prosa livre e elimina por construção a chamada extra de LLM do *Feedback Filter*.
2. **Roteamento Triplo Calibrado (*Accept / Rewrite / Escalate*):**
   - **Fast Accept ($p \ge \tau_{\text{accept}}$):** Aceita imediatamente rascunhos conformes após a 1ª geração, evitando *over-editing* e custo desnecessário.
   - **Guided Rewrite ($p < \tau_{\text{reject}}$):** Converte falhas claras do checklist em instruções determinísticas de reparo para o LLM gerador.
   - **Uncertainty Escalation ($\tau_{\text{reject}} \le p < \tau_{\text{accept}}$):** Isola requisitos complexos/ambíguos para revisão humana especializada.

---

## Estrutura do Repositório

```text
.
├── data/zenodo_rosskothen2026/   # Pacote oficial (249 reqs fonte, 410 MASTeR humanos, scores de referência)
├── prompts/                      # Prompts Jinja2 oficiais de Roßkothen et al. (preservados para validade interna)
├── src/req_jev/                  # Pacote Python principal
│   ├── schemas.py                # Contratos Pydantic, resultados JEV e telemetria de custo/latência
│   ├── dataset.py                # Loader dos 249 requisitos (FLEX, ECSS, CSE, TSS, EVS)
│   ├── prompts_loader.py         # Renderização Jinja2 e tabela de custos USD por modelo
│   ├── llm_backends.py           # Adaptadores assíncronos para Google Gemini e OpenRouter (Qwen3-235B)
│   ├── jev_validator.py          # Validador System One (typesafe-sdk) e roteador cascata calibrado
│   ├── pipelines.py              # Pipelines controlados: oneshot, llm_judge e jev_cascade
│   └── evaluation.py             # Métricas MASTeR F1, EMD, BLEU, métricas operacionais e testes estatísticos
├── scripts/
│   ├── analyze_baseline_paper.py       # Extrai tabelas e métricas de referência do artigo original
│   ├── run_experiment.py               # Executor assíncrono com retomada automática em JSONL
│   └── evaluate_and_export_latex.py    # Consolida resultados e exporta tabelas LaTeX para paper/tables/
├── tests/test_pipeline_smoke.py  # Testes automatizados offline dos pipelines e métricas
└── paper/                        # Artigo completo em LaTeX (IEEEtran em Português) pronto para Overleaf
```

---

## Configuração Rápida

### 1. Ambiente virtual e dependências

O ambiente `.venv` já está configurado com Python 3.13. Caso precise recriar:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Configurar chaves de API

Copie o arquivo `.env.example` para `.env` e preencha suas chaves:

```bash
cp .env.example .env
```

- `TYPESAFE_API_KEY`: chave oficial da API TypeSafe AI (`jev-latest`).
- `GEMINI_API_KEY`: chave do Google AI Studio (`gemini-2.5-flash`).
- `OPENROUTER_API_KEY` *(opcional, para replicação 1:1 com `qwen/qwen3-235b-a22b-2507`)*: chave do OpenRouter.

---

## Executando os Experimentos

### Passo 1 — Smoke test gratuito (offline)
```bash
.venv/bin/pytest -v
```

### Passo 2 — Teste piloto em 5 requisitos reais
```bash
.venv/bin/python scripts/run_experiment.py --provider gemini --mode jev_cascade --limit 5
```

### Passo 3 — Bateria completa nos 249 requisitos (suporta pausa e retomada automática)
```bash
# Bloco A: Família Gemini (controle interno completo nos 3 modos)
.venv/bin/python scripts/run_experiment.py --provider gemini --mode oneshot
.venv/bin/python scripts/run_experiment.py --provider gemini --mode llm_judge
.venv/bin/python scripts/run_experiment.py --provider gemini --mode jev_cascade

# Bloco B: Qwen3-235B via OpenRouter (comparação 1:1 com Roßkothen et al. 2026)
.venv/bin/python scripts/run_experiment.py --provider openrouter --mode jev_cascade
```

### Passo 4 — Avaliar resultados e gerar tabelas LaTeX para o artigo
```bash
.venv/bin/python scripts/evaluate_and_export_latex.py
```
Isso gerará `results/summary_metrics.csv` e `paper/tables/experimental_results_auto.tex`, pronto para compilar no Overleaf!
