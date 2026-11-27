---
layout: blog-post
title: "We Need to Monitor to Survive"
description: "A broader view of monitoring research with some historical motivation and faith in humankind."
date: 2026-11-27 03:09:00 +0300
draft: false
featured: false
related_posts: false
toc:
  beginning: false
---
We are going through exciting times, with AI developing at an incredible pace. It can one-shot production-grade applications and, in just weeks, crack [math problems](https://en.wikipedia.org/wiki/Alp%C3%B6ge%27s_proof_of_the_Riemann_hypothesis) that humans couldn't solve for over 150 years. All of these breakthroughs fuel concerns about catastrophic and existential risks from AI. Many prominent voices in the AI community hold varying [P(doom)](https://en.wikipedia.org/wiki/P(doom)) estimates, from [Geoffrey Hinton's 10–20%](https://www.youtube.com/watch?v=PTF5Up1hMhw&t=2283s) to [David Krueger's 75%](https://www.youtube.com/watch?v=PyBw-dIcKW4). If you aren't a big believer in existential risks, there are still [catastrophic risks](https://safe.ai/ai-risk), which are perhaps more plausible and more imminent. So, one way or another, there is a significant risk that we're _cooked_.

Still, I'm very optimistic when it comes to existential risks because I strongly believe in our ability to survive as a species. _Homo sapiens_ has managed to survive for 300,000 years. Even if we narrow it down to written history—just the last 6,000 years, packed with wars, death, and disaster—we've learned a lot about survival. While it's impossible to squeeze history into one concept, I think there is a major factor that helped us survive for millennia and could once again carry us through arguably _the biggest challenge of the century_: **monitoring**.

<figure>
  <img src="{{ '/assets/img/blog/monitor-to-survive/egyptian_monitoring.png' | relative_url }}"
       alt="A moonless night on the Nile, with the sky opening onto a distant digital city and a faint trace of its light on the river.">
  <figcaption></figcaption>
</figure>

## We survived because we monitored.

The history of monitoring goes back millions of years. _Homo erectus_ stood up, which let them better monitor their surroundings, prey, and threats. People in Ancient Egypt used structures called nilometers to monitor the Nile's water levels and predict harvests based on floods and droughts. They also monitored the stars to track when the Nile would begin to flood. Considering that economic activity was centered around the Nile, it's easy to see how crucial monitoring was to Ancient Egyptian survival.

Imagine a medieval European city with its must-have fortified walls and towers. Those watchtowers might be one of the first distributed, scalable monitoring systems humans ever built. A network of watchtowers placed within line of sight was crucial for flagging incoming threats in advance. This relay mechanism allowed city residents to spot threats kilometers away, extending monitoring beyond immediate physical perception.

One of the most extreme examples of monitoring has to be intelligence work. A defining feature of any powerful state is its domestic and international intelligence capabilities. Perhaps the most famous example is Allied cryptanalysts cracking the Enigma code—a breakthrough estimated to have shortened World War II by two years, saving millions of lives. Beyond that, intelligence has played a key role in asserting influence across complex domestic and international power dynamics.

Beyond these historical examples, today we _monitor_ everything. We _monitor_ every machine we build—its energy consumption, temperature, speed, acceleration, and pretty much any other physical property you can think of. Every piece of hardware and software we build is packed with telemetry so we can _monitor_ it. We _monitor_ our cells and our blood to protect our health and cure yet-incurable diseases. We _monitor_ seismic activity to track and forecast earthquakes. We _monitor_ the markets for our investments. Sometimes we can't resist the joy of monitoring—we even [_monitor_ the situation](https://knowyourmeme.com/memes/monitoring-the-situation). I don't think it's absurd to say we are a civilization of monitoring. We live inside a huge _superpanopticon_ (Poster, 1990). Given all that, why can't we overcome the grand survival challenge of the century through monitoring?



## A Holistic View of Monitoring

Given my historically rooted motivation, I'm invested in monitoring research as one of the most important technical tools against existential risk. In AI safety, the term is usually understood as Chain-of-Thought (CoT) or output monitoring, where another LLM (smaller or larger than the target model) or a representation-based classifier flags outputs for a specific property, like reward hacking or harmful responses. However, I view monitoring as a broader umbrella covering a variety of approaches. We can categorize it into three levels: (1) immediate monitoring, (2) broad monitoring, and (3) predictive monitoring.

### 1. Immediate Monitoring

This is monitoring in the conventional sense, which splits into two branches: 

- **Black-box (natural-language-based) monitoring** inspects the model's reasoning traces or full outputs using another LLM ([Korbak et al., 2025](https://arxiv.org/abs/2507.11473); [Baker et al., 2025](https://arxiv.org/abs/2503.11926); [Guan et al., 2025](https://arxiv.org/abs/2512.18311)). While LLM monitors scale well with test-time compute, another way to improve black-box monitoring is to make reasoning traces more monitorable in the first place. Since these traces don't always faithfully reflect a model's true reasoning ([Turpin et al., 2023](https://arxiv.org/abs/2305.04388); [Lanham et al., 2023](https://arxiv.org/abs/2307.13702); [Chen et al., 2025](https://arxiv.org/abs/2505.05410); [Barez et al., 2026](https://arxiv.org/abs/2611.31271)), misalignment can go undetected regardless of how capable the monitor is. Building on efforts to improve CoT faithfulness, [Hase and Potts (2026)](https://arxiv.org/abs/2602.20710) show that optimizing for faithfulness directly improves monitorability.

- **White-box (representation-based) monitoring** inspects a model's internal representations via probes, offering a cheaper alternative for production-grade systems.

Regardless of the approach, using immediate monitors during training is risky, as they can incentivize models to obfuscate their CoTs ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)) or representations. [Recent work from Goodfire](https://www.goodfire.com/research/mechanistic-shaping) suggests that linear probes periodically kept on-policy show promise in this setting.

### 2. Broad Monitoring

Immediate monitoring has a narrow focus on individual examples, which is only useful as long as our monitors are precise. One way to hedge against the risk of relying on single-example flags is to monitor from a broader perspective. When we take a step back, we can monitor multiple instances together instead of one at a time. For example, [recent work](https://arxiv.org/abs/2607.07368) shows that distributing a DPO data-poisoning attack across multiple agents can evade monitors that easily catch a single agent. Or think of a training setup where a reward model has some quirky bias, like rewarding specific formatting (such as bullet lists) or the use of a particular unsafe function call. Unless a monitor LLM is explicitly prompted to look for these subtle biases—or already treats them as unsafe or cheating—they rarely stand out in a single example. However, cross-sample analysis can catch such anomalies.

Another dimension of anomaly detection is monitoring the environment the agents run in. Think of a [rogue agent that escapes a sandbox](https://openai.com/index/hugging-face-model-evaluation-security-incident/) assumed to have no web access, or agents [overwhelming Wikimedia's servers](https://wikimediafoundation.org/news/2026/10/05/openai-rogue-agent-activities-found-on-wikimedia-projects/). Even if agents evade CoT or output monitors, a system-level perspective adds an extra layer of safety ([Zheng et al., 2025](https://arxiv.org/abs/2508.02736)). This could include tracking network traffic or unusual I/O activity (e.g., [agents leaving messages for others](https://cdn.openai.com/pdf/67869394-cb91-4c12-888c-5cbd85c7814c/OpenAI-Hugging-Face%20Incident-Technical-Report.pdf)).

Stepping back even further, we can move beyond monitoring for misaligned behavior at runtime to monitoring the misalignment tendencies of the model itself—making the model the unit of granularity. This involves intermediate safety evaluations, red-teaming, and pre-deployment audits ([OpenAI: CoT Monitorability](https://openai.com/safety-alignment/cot-monitorability-model-lifecycle/); [OpenAI: Managing Frontier AI Risk](https://openai.com/index/managing-frontier-ai-risk/)). 


### 3. Predictive Monitoring

While all of the previous approaches focus on runtime monitoring or evaluations throughout the model development cycle, we also need a predictive perspective—one that helps us infer trends and make forecasts before training even begins. This perspective can involve proxy metrics or tasks that have a predictive or causal relationship to model monitorability. 

CoT controllability is a good example. It relies on the premise that if a model can control its CoT, it can evade CoT monitors once it has evaluation awareness. [Yong et al. (2026)](https://arxiv.org/abs/2611.21799) show that high controllability leads to lower monitorability; they induce high CoT controllability by training models on CoTs rewritten under unique personas and stylistic constraints.

Another useful proxy is a model's no-CoT performance. Previous work shows that CoTs are likely to verbalize reasoning only when it is necessary to solve the task ([Emmons et al., 2025](https://arxiv.org/abs/2507.05246)). However, as models get more capable, tasks that once required extra inference-time compute can be solved in a single forward pass. That eliminates the need for verbose CoTs—and for spelling out the key factors driving their reasoning. Meanwhile, looped transformers ([Geiping et al., 2025](https://arxiv.org/abs/2502.05171); [Ouro](https://arxiv.org/abs/2510.25741); [Fan et al., 2026](https://arxiv.org/abs/2606.31779)) push this further by enabling multi-step reasoning entirely in latent space. ThinkFast ([Gould et al., 2026](https://arxiv.org/abs/2606.07157)) provides a benchmark for tracking task-completion time horizons with no CoT tokens, showing that models' latent reasoning capabilities double annually. 

Both CoT controllability and no-CoT capabilities serve as predictors of future model monitorability and broader trends, rather than just capturing the current state. For instance, rising no-CoT capabilities call for developing better latent monitoring approaches.

Beyond these proxies, we need approaches that can predict the safety-related behaviors of a proposed training pipeline—for instance, how different pretraining data mixtures or RL environments affect monitorability, sandbagging, or situational awareness. Can we come up with a generalized framework, or even a kind of scaling law, for these types of behaviors? As we move toward Recursive Self-Improvement (RSI), such approaches will become even more crucial.

## _Monitoring_ Ahead and Further Reading

I'm hyped for new research across all of these fronts—there is still so much to build! While I didn't plan this blog post as an exhaustive list of references for monitoring research, I think the resources I cited throughout the text are good starting points. Apart from those, [Peter](https://peterbhase.github.io/) and [Chris](https://stanford.edu/~cgpotts/) previously wrote a [great post](https://ai.stanford.edu/blog/cot-monitoring-history/) covering the short history of CoT monitoring. 

Also, shoutout to [NDIF's Looped Lens Challenge](https://ndif.us/looped-lens-challenge/), aimed at uncovering looped reasoning of [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/).


**Disclaimer:** _Gemini and Claude models were used to polish the content and the illustration was generated with Gemini._