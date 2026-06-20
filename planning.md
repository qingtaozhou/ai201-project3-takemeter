# Project 3 Planning: Space-Stock Discourse Classifier

## Goal

This project will build a classifier for posts and comments from a public space-stock investing community. The model will not try to predict whether a stock is a good investment. Instead, it will classify the kind of discourse a post contributes: evidence-based investment analysis, factual information sharing, speculative hype, or risk-focused skepticism.

## Community

I will study public Stocktwits streams for space-related stocks and ETFs, centered on ASTS and related tickers such as RKLB, LUNR, SPCE, ARKX, UFO, NASA, and SpaceX/SPCX discussion. This community is a good fit because it is active, text-heavy, and centered on investment theses that depend on uncertain future events: satellite deployment, launch schedules, carrier partnerships, FCC/regulatory progress, SpaceX/Starlink competition, financing, NASA or government-space news, and space-sector ETFs.

The discourse is varied enough for classification because users post many different kinds of content. Some posts are detailed due diligence using filings, technical details, or valuation assumptions. Others simply share news links, price movement, analyst headlines, or ETF holdings. Some posts are bullish hype about space stocks becoming huge winners, while others warn about dilution, execution risk, competition, or lack of current revenue. These distinctions matter because investors in the community are trying to separate useful reasoning from excitement, fear, and headline-chasing.

## Labels

### evidence_analysis

Definition: The post makes a structured investment argument using specific evidence, such as financial metrics, contract details, technical milestones, launch timelines, regulatory filings, customer agreements, valuation comparisons, or competitive analysis.

Example posts:

- "ASTS's upside depends less on today's revenue and more on whether the first commercial satellites prove enough capacity for partner carriers. If each satellite can support the expected number of beams and the company avoids another major delay, the current valuation could be justified, but the dilution risk still matters."
- "Starlink Direct-to-Cell is not automatically the same product as ASTS. The key comparison is spectrum access, carrier relationships, handset compatibility, satellite size, and service quality. SpaceX has launch scale, but ASTS may have an advantage if its carrier agreements turn into real commercial coverage."

### information_update

Definition: The post primarily shares a factual update, link, filing, quote, launch item, price movement, ETF holding, analyst note, or government/industry news without adding a developed original argument.

Example posts:

- "AST SpaceMobile filed a new 8-K today. Link in the comments."
- "ARKX increased its ASTS position this week, while another space ETF reduced exposure to small-cap satellite names."

### speculative_hype

Definition: The post makes an optimistic or dramatic claim about future stock performance, company success, short squeezes, buyout potential, or industry disruption with little evidence or mostly emotional reasoning.

Example posts:

- "ASTS is going to be the next trillion-dollar space stock once people realize phones will connect directly to satellites."
- "NASA and SpaceX are both focused on space communications, so ASTS is obviously in the right place at the right time. This is going to $100."

### risk_skepticism

Definition: The post focuses on downside risk, uncertainty, or reasons the investment thesis may fail, using either concrete evidence or a clearly stated concern about financing, execution, competition, regulation, valuation, or technology.

Example posts:

- "The bull case ignores how much capital ASTS may still need before full deployment. If the company raises money again, current shareholders could face meaningful dilution."
- "SpaceX/Starlink has launch scale, engineering depth, and a strong brand. Even if ASTS technology works, competition could reduce pricing power or slow carrier adoption."

## Hard Edge Cases

The hardest edge cases will be posts that contain one piece of evidence but use it mainly to support a broad emotional claim. For example: "ASTS has agreements with major carriers, so once the satellites are up this could easily be a $100 stock." This could look like `evidence_analysis` because it mentions carrier agreements, but it could also be `speculative_hype` because it jumps to a price target without showing share count, revenue assumptions, margins, timeline, or probability of execution.

My annotation rule is to label a post by its primary contribution. A post is `evidence_analysis` only when the evidence is specific, relevant, and used in a chain of reasoning. A post is `speculative_hype` when evidence is thin, decorative, or used mainly to create excitement. A post is `information_update` when the poster is mostly relaying a link, filing, quote, launch update, ETF holding, or price move. A post is `risk_skepticism` when its main purpose is to warn about downside, even if it includes some evidence.

For ambiguous posts, I will write a short note in the annotation file explaining the competing labels and final decision. If the same ambiguity appears repeatedly, I will update this planning document before labeling the rest of the dataset so that the rule stays consistent.

Specific difficult cases from annotation:

- A post argued that ASTS would be affected by a SpaceX IPO because space ETFs would rebalance away from ASTS. It could be `risk_skepticism` because it predicts downside, but I labeled it `evidence_analysis` because the post gave a specific mechanism: ETF weights, SpaceX inclusion, and likely forced reduction.
- A post said ASTS had solved recent satellite stacking/launch issues and was in the running for a Japan contract. It could be `speculative_hype` because it was optimistic, but I labeled it `evidence_analysis` because it listed concrete milestones and open questions instead of only making an upside claim.
- A post said ASTS had "no service" and that carrier agreements were not true contracts. It could be `evidence_analysis` because it mentioned agreement terms and carrier relationships, but I labeled it `risk_skepticism` because the main purpose was to warn that the bull thesis was overstated.
- A post shared a SpaceX IPO headline with a dramatic valuation number. It could look like `speculative_hype`, but I labeled it `information_update` when the poster mainly relayed an outside news item rather than adding their own investment argument.

## Data Collection Plan

I collected 200 public Stocktwits posts from space-stock streams, including ASTS, RKLB, LUNR, SPCE, ARKX, and UFO. The dataset is saved as `data/labeled_examples.csv`. I sampled across company-news posts, launch updates, price-movement posts, SpaceX/Starlink competitor posts, government or NASA-related posts, ETF/space-sector posts, and longer due-diligence discussions.

Final label distribution:

- `evidence_analysis`: 40 examples
- `information_update`: 60 examples
- `speculative_hype`: 60 examples
- `risk_skepticism`: 40 examples

No label accounts for more than 70% of the dataset. Evidence-based analysis and risk-focused skepticism were less common than news sharing and hype, so I sampled more targeted long posts containing terms such as revenue, launch, contract, dilution, SpaceX, ETF, and satellite before finalizing the CSV.

## Evaluation Metrics

Accuracy alone is not enough because the dataset may be imbalanced and because some mistakes are more harmful than others. For example, a model that labels most posts as `information_update` could have decent accuracy if news posts are common, but it would not be useful for finding real analysis or separating hype from risk.

I will use these metrics:

- Overall accuracy, to measure the basic percentage of correct predictions.
- Macro F1, to give each label equal importance even if some labels are less common.
- Per-label precision and recall, especially for `evidence_analysis` and `speculative_hype`.
- Confusion matrix, to see which labels the model mixes up most often.

Macro F1 is the main metric because all four labels matter. Per-label recall for `evidence_analysis` matters because a useful community tool should not miss most high-quality analysis. Per-label precision for `evidence_analysis` also matters because users would lose trust if hype posts were frequently promoted as analysis. For `speculative_hype`, precision matters because falsely calling cautious or informational posts hype could unfairly dismiss useful discussion.

## Definition of Success

A genuinely useful classifier should reach at least 0.75 macro F1 on a held-out test set, with no individual label below 0.65 F1. For `evidence_analysis`, I want at least 0.75 precision and 0.70 recall, because the tool should surface analysis without flooding users with unsupported claims. For `speculative_hype`, I want at least 0.70 precision so the model does not over-accuse ordinary bullish posts.

For a real community tool, I would consider the first version good enough for limited deployment if it reaches at least 0.70 macro F1, no label falls below 0.60 F1, and manual review of 50 recent predictions shows that the most visible errors are understandable edge cases rather than obvious misclassifications. I would not use the classifier for moderation or automatic removal. It would be appropriate only for softer uses like filtering, tagging, summarizing discussion quality, or helping users find due-diligence posts.

## AI Tool Plan

### Label stress-testing

Before annotating the full dataset, I will give an AI tool my four label definitions and ask it to generate 5-10 borderline space-stock posts. The prompt will specifically ask for posts that sit between `evidence_analysis` and `speculative_hype`, between `information_update` and `evidence_analysis`, and between `risk_skepticism` and `evidence_analysis`.

I will manually classify each generated edge case using the rules in this document. If I cannot classify several of them cleanly, I will tighten the definitions before collecting the final 200 examples. I will not include AI-generated posts in the training or test data; they are only for stress-testing the taxonomy.

### Annotation assistance

I used Codex as annotation assistance to create an initial labeled CSV from public Stocktwits posts using the label rules in this document. The CSV keeps a `notes` column for difficult cases and a `source_url` column for provenance. Before final submission, any assisted labels should be reviewed against the definitions rather than accepted blindly.

If the LLM frequently disagrees with me on a label, I will not automatically trust either answer. I will review those disagreements as possible signs that the taxonomy needs a better edge-case rule.

### Failure analysis

After evaluating the model, I will give an AI tool a list of wrong predictions with the post text, true label, predicted label, and my brief notes. I will ask it to identify repeated error patterns, such as hype being mistaken for analysis when a post includes one statistic, or risk-focused analysis being confused with general skepticism.

I will verify any AI-suggested pattern myself by checking the actual examples and the confusion matrix. The AI tool can help generate hypotheses, but I will only report patterns that are supported by the model's errors and my manual review.
