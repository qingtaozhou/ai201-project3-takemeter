# Space-Stock Discourse Classifier

This project classifies public space-stock investing posts by discourse type. The goal is not to predict whether ASTS, SpaceX/SPCX, NASA-themed ETFs, or other space stocks are good investments. The goal is to classify what kind of contribution a post makes: factual update, evidence-based analysis, speculative hype, or risk-focused skepticism.

## Community And Data

I collected public Stocktwits posts from space-related streams centered on ASTS and related tickers such as RKLB, LUNR, SPCE, ARKX, UFO, NASA, and SpaceX/SPCX discussion. This community works well for classification because the same subject can produce very different discourse: launch news, ETF/IPO updates, contract speculation, valuation arguments, bullish hype, and warnings about dilution or competition.

The dataset is [data/labeled_examples.csv](data/labeled_examples.csv). It contains 200 labeled examples with `text`, `label`, `notes`, `source_url`, `created_at`, and `symbols`.

| Label | Count |
|---|---:|
| `evidence_analysis` | 40 |
| `information_update` | 60 |
| `speculative_hype` | 60 |
| `risk_skepticism` | 40 |

No label is above 70% of the dataset; the largest labels are 30% each.

## Label Definitions

| Label | Definition |
|---|---|
| `evidence_analysis` | A structured investment argument using specific evidence such as financial metrics, launch timelines, technical milestones, contract details, valuation comparisons, or competitive analysis. |
| `information_update` | A factual update, link, filing, quote, launch item, analyst note, ETF holding, or news item without a developed original argument. |
| `speculative_hype` | An optimistic or dramatic claim about future price movement, company success, short squeezes, buyouts, or disruption with little evidence or mostly emotional reasoning. |
| `risk_skepticism` | A post focused on downside risk or reasons the investment thesis may fail, including financing, execution, competition, regulation, valuation, or technology concerns. |

The hardest boundary is between `evidence_analysis` and `speculative_hype`. Many posts contain one real detail, such as a contract, launch, ETF rebalance, or valuation number, but use it mainly to justify an unsupported price target or emotional claim.

## Models

The baseline model and the fine-tuned model were evaluated on a 30-example held-out test set. The fine-tuned model is `distilbert-base-uncased`.

| Model | Accuracy |
|---|---:|
| Baseline | 0.7667 |
| Fine-tuned DistilBERT | 0.7667 |

The fine-tuned model matched the baseline overall. That is good enough to show that fine-tuning did not break the task, but it did not improve over the baseline.

## Evaluation

Fine-tuned per-class metrics, reconstructed from the saved confusion matrix:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `evidence_analysis` | 0.7500 | 0.5000 | 0.6000 | 6 |
| `information_update` | 0.8000 | 0.8889 | 0.8421 | 9 |
| `speculative_hype` | 0.6667 | 0.6667 | 0.6667 | 9 |
| `risk_skepticism` | 0.8571 | 1.0000 | 0.9231 | 6 |
| **Macro average** | | | **0.7580** | 30 |

Baseline per-class metrics were not saved in the current repo artifacts, so I can report the saved baseline accuracy but not the full baseline classification report without rerunning evaluation.

Fine-tuned confusion matrix:

Rows are true labels; columns are predicted labels.

| True \ Predicted | `evidence_analysis` | `information_update` | `speculative_hype` | `risk_skepticism` |
|---|---:|---:|---:|---:|
| `evidence_analysis` | 3 | 1 | 2 | 0 |
| `information_update` | 0 | 8 | 1 | 0 |
| `speculative_hype` | 1 | 1 | 6 | 1 |
| `risk_skepticism` | 0 | 0 | 0 | 6 |

The strongest class is `risk_skepticism`: all 6 true risk examples were identified correctly. `information_update` is also strong, with 8 of 9 correct. The weakest class is `evidence_analysis`, where only 3 of 6 examples were correct. Most remaining errors are boundary errors between analysis, hype, and updates.

## Failure Analysis

I used Codex to inspect the confusion matrix and surface likely error patterns, then checked those patterns against the label definitions and the dataset examples. I kept the patterns that were supported by the matrix and discarded the earlier interpretation from an older run, where the model appeared to collapse into `information_update`.

The main confusion is `evidence_analysis` being predicted as either `speculative_hype` or `information_update`. This is exactly the hardest label boundary from the spec. In this domain, a post can mention launches, contracts, SpaceX, ETFs, revenue, or valuation and still be hype if it does not connect those facts into a real argument.

Specific wrong-prediction patterns:

| True Label | Predicted Label | Example Excerpt | Why It Failed |
|---|---|---|---|
| `evidence_analysis` | `speculative_hype` | "ASTS will go down with SpaceX IPO as these ETF will include SpaceX and need to rebalance..." | The post makes a concrete ETF-rebalancing argument, but the directional price claim may look like ordinary stock hype. The model likely overweighted the confident price direction and ticker-heavy style. |
| `evidence_analysis` | `information_update` | "They solved stacking for block2, put 3 new sats into orbit, are essentially ready to ship the next 3..." | This is not just a launch update; it uses milestones to reason about execution risk. The model likely saw factual launch language and treated it as a news update. |
| `speculative_hype` | `evidence_analysis` | "SpaceX could be worth $10-$30 trillion in a decade or two..." | The post uses comparison and valuation language, but the conclusion is too large and unsupported. The model likely mistook valuation vocabulary for evidence-based reasoning. |

These errors are more about boundary difficulty than random failure. The labels are conceptually meaningful, but the dataset needs more paired examples where the topic is similar and only the discourse function changes. For example, several posts about a SpaceX IPO should show all four labels: a headline-only update, a reasoned ETF-rebalance analysis, a bullish moonshot post, and a skeptical overvaluation warning.

## Sample Classifications

The saved artifacts do not include per-example confidence scores, so the table below uses representative examples and the expected label behavior. To fully satisfy a production-style report, the notebook should save a `sample_predictions.csv` with text, predicted label, true label, and confidence.

| Example Post Excerpt | Expected Label | Comment |
|---|---|---|
| "SpaceX Finalizes Blockbuster IPO Price at $135 Per Share..." | `information_update` | A prediction of `information_update` is reasonable because the post mainly relays a news headline. |
| "There is no contract for service. There is no service..." | `risk_skepticism` | A prediction of `risk_skepticism` is reasonable because the post challenges the ASTS bull thesis. |
| "SpaceX could be worth $10-$30 trillion..." | `speculative_hype` | This should be hype unless the post supports the valuation with detailed assumptions. |
| "Space ETFs will rebalance because SpaceX inclusion changes index weights..." | `evidence_analysis` | This should be analysis because it gives a concrete mechanism rather than only a price claim. |

## Reflection

The model captured some of the intended structure. It learned `risk_skepticism` well and generally recognized factual news/update posts. That means the labels are not arbitrary: the model can find stable patterns when the discourse function is clear.

The gap is in mixed posts. The model still struggles when posts combine facts with hype or facts with analysis. It appears to rely partly on topic words like SpaceX, launch, ETF, contract, revenue, and valuation. Those words are useful, but they are not enough. The intended boundary depends on how the post uses evidence, not just whether evidence-like language appears.

## Spec Reflection

The spec helped guide implementation by forcing the label definitions and edge-case rules before annotation. The rule "primary contribution decides the label" was especially useful for posts that mentioned real facts but mostly served as hype or skepticism.

The implementation diverged from the original plan in the data source. I initially planned to use Reddit's r/ASTSpaceMobile, but public Reddit JSON access was blocked during collection. I switched to public Stocktwits space-stock streams so the dataset could still focus on ASTS, SpaceX, NASA/ETF discussion, and public investor discourse.

## AI Usage

I used AI assistance in several specific ways:

1. Planning and taxonomy drafting: I directed Codex to turn the milestone requirements into a concrete `planning.md` for a space-stock community. Codex produced draft labels and edge-case rules. I narrowed the topic toward ASTS, SpaceX, NASA, and ETF discussion.
2. Annotation assistance: I directed Codex to collect public posts and create an initial labeled CSV using the definitions in `planning.md`. Codex produced [data/labeled_examples.csv](data/labeled_examples.csv) with labels, notes, and source URLs. I reviewed distribution and edge-case notes, and the project discloses that annotation was AI-assisted.
3. Failure analysis: I directed Codex to inspect the saved evaluation output and confusion matrix. Codex reconstructed fine-tuned per-class metrics from the matrix and identified the main error boundary: analysis versus hype/update. I corrected the README after a newer evaluation run replaced the earlier confusion matrix.

## Files

| File | Purpose |
|---|---|
| [planning.md](planning.md) | Working design notes, labels, edge cases, data plan, metrics plan, and AI tool plan. |
| [data/labeled_examples.csv](data/labeled_examples.csv) | Single labeled dataset used for model training and evaluation. |
| [evaluation_results.json](evaluation_results.json) | Saved aggregate evaluation results. |
| [confusion_matrix.png](confusion_matrix.png) | Fine-tuned model confusion matrix image. |
