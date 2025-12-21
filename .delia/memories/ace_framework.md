# ACE: Agentic Context Engineering

## Abstract
Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation—modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

## 1 Introduction
Modern AI applications based on large language models (LLMs), such as LLM agents [49, 52] and compound AI systems [55], increasingly depend on context adaptation. Instead of modifying model weights, context adaptation improves performance after model training by incorporating clarified instructions, structured reasoning steps, or domain-specific input formats directly into the model’s inputs. Contexts underpin many AI system components, including system prompts that guide downstream tasks [4, 36], memory that carries past facts and experiences [41, 48], and factual evidence that reduces hallucination and supplements knowledge [6].

Adapting through contexts rather than weights offers several key advantages. Contexts are interpretable and explainable for users and developers [45, 47], allow rapid integration of new knowledge at runtime [7, 27], and can be shared across models or modules in a compound system [23]. Meanwhile, advances in long-context LLMs [39] and context-efficient inference such as KV cache reuse [17, 51] are making context-based approaches increasingly practical for deployment. As a result, context adaptation is emerging as a central paradigm for building capable, scalable, and self-improving AI systems.

Despite this progress, existing approaches to context adaptation face two key limitations. First, a brevity bias: many prompt optimizers prioritize concise, broadly applicable instructions over comprehensive accumulation. For example, GEPA [4] highlights brevity as a strength, but such abstraction can omit domain-specific heuristics, tool-use guidelines, or common failure modes that matter in practice [16]. This objective aligns with validation metrics in some settings, but often fails to capture the detailed strategies required by agents and knowledge-intensive applications. Second, context collapse: methods that rely on monolithic rewriting by an LLM often degrade into shorter, less informative summaries over time, causing sharp performance declines. In domains such as interactive agents [38, 43, 57], domain-specific programming [53, 56], and financial or legal analysis [18, 33, 44], strong performance depends on retaining detailed, task-specific knowledge rather than compressing it away.

## 2 Background and Motivation
### 2.1 Context Adaptation
Context adaptation (or context engineering) refers to methods that improve model behavior by constructing or modifying inputs to an LLM, rather than altering its weights. The current state of the art leverages natural language feedback [4, 40, 54]. In this paradigm, a language model inspects the current context along with signals such as execution traces, reasoning steps, or validation results, and generates natural language feedback on how the context should be revised. 

### 2.2 Limitations of Existing Context Adaptation Methods
**The Brevity Bias.** A recurring limitation of context adaptation methods is brevity bias: the tendency of optimization to collapse toward short, generic prompts. 

**Context Collapse.** Monolithic rewriting of context by an LLM can collapse it into shorter, less informative summaries, leading to sharp performance drops. As the context grows large, the model tends to compress it into much shorter, less informative summaries, causing a dramatic loss of information.

## 3 Agentic Context Engineering (ACE)
We present ACE (Agentic Context Engineering), a framework for scalable and efficient context adaptation in both offline and online scenarios. ACE treats contexts as evolving playbooks that continuously accumulate, refine, and organize strategies over time. Building on the agentic design of Dynamic Cheatsheet [41], ACE introduces a structured division of labor across three roles:
1. **The Generator:** produces reasoning trajectories.
2. **The Reflector:** distills concrete insights from successes and errors.
3. **The Curator:** integrates these insights into structured context updates.

ACE introduces three key innovations:
1. A dedicated **Reflector** that separates evaluation and insight extraction from curation.
2. **Incremental delta updates** that replace costly monolithic rewrites with localized edits.
3. A **grow-and-refine** mechanism that balances steady context expansion with redundancy control.

### 3.1 Incremental Delta Updates
A core design principle of ACE is to represent context as a collection of structured, itemized bullets, rather than a single monolithic prompt. Each bullet consists of metadata (unique identifier, helpful/harmful counters) and content (reusable strategy, domain concept, or failure mode).

### 3.2 Grow-and-Refine
ACE ensures that contexts remain compact and relevant through periodic or lazy refinement. In grow-and-refine, bullets with new identifiers are appended, while existing bullets are updated in place. A de-duplication step prunes redundancy by comparing bullets via semantic embeddings.

## 4 Results
ACE consistently outperforms strong baselines, yielding average gains of 10.6% on agents and 8.6% on domain-specific benchmarks. ACE is able to construct effective contexts without labeled supervision, leveraging execution feedback. ACE requires significantly fewer rollouts and achieves 86.9% lower adaptation latency than existing adaptive methods.

## 5 Discussion
**Longer Context != Higher Serving Cost.** Techniques such as KV cache reuse, compression, and offload allow frequently reused context segments to be cached, avoiding repetitive prefill operations.
**Implications for Online and Continuous Learning.** ACE offers a flexible and efficient alternative to conventional model fine-tuning. Because contexts are human-interpretable, ACE enables selective unlearning.

## A Related Work on Agent Memory
A growing body of work explores how agents can accumulate experience (AgentFly, AWM, A-MEM, Agentic Plan Caching). ACE differs by tackling the broader challenge of context adaptation spanning system prompts, factual evidence, and agent memory.

## B Limitations and Challenges
ACE reliance on a strong Reflector: if the Reflector fails, the context may become noisy. ACE is most beneficial in settings that demand detailed domain knowledge or complex tool use.

## C AppWorld Leaderboard Snapshot (09/2025)
ACE (59.4%) matches top-ranked production-level agents on average and surpasses them on harder challenge splits.

## D Prompts
[Various prompt templates for Generator, Reflector, and Curator are included in the original text, focusing on structured JSON outputs, chain-of-thought reasoning, and itemized delta updates.]

### Figure 6: ICL-baseline Generator prompt on AppWorld
[Prompt structure for autonomous interaction with apps using APIs, including instructions for REPL environment and API documentation usage.]

### Figure 7: Dynamic Cheatsheet Generator prompt on AppWorld
[Prompt structure incorporating a 'CHEATSHEET' section for strategies and patterns.]

### Figure 8: GEPA prompt on AppWorld
[Prompt structure with domain-specific strategies for various tasks (bill splitting, file organization, etc.).]

### Figure 9: ACE Generator prompt on AppWorld
[Prompt structure incorporating an 'ACE Playbook' section.]

### Figure 10: ACE Reflector prompt on AppWorld
[Expert diagnosis prompt to identify errors, root causes, and correct approaches, outputting structured JSON.]

### Figure 11: ACE Curator prompt on AppWorld
[Knowledge curation prompt to integrate new insights into the playbook as incremental ADD operations.]

### Figure 12: ACE Generator prompt on FINER
[Specific generator prompt for financial analysis tasks using the playbook.]

### Figure 13: ACE Reflector prompt on FINER
[Specific reflector prompt for financial analysis, including bullet-point tagging for helpfulness/harmfulness.]
