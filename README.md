# KPSC

KPSC has two methods: KPSC-P and KPSC-F

## Conference Version
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://dl.acm.org/doi/10.1145/3503161.3548004)

Prompt-based Zero-shot Video Moment Retrieval

ACM International Conference on Multimedia (**ACM Multimedia**), 2022

### Abstract
> Video moment retrieval aims at localizing a specific moment from an untrimmed video by a sentence query. Most methods rely on heavy annotations of video moment-query pairs. Recent zero-shot methods reduced annotation cost, yet they neglected the global visual feature due to the separation of video and text learning process. To avoid the lack of visual features, we propose a Prompt-based Zero-shot Video Moment Retrieval (PZVMR) method. Motivated by the frame of prompt learning, we design two modules: 1) Proposal Prompt (PP): We randomly masks sequential frames to build a prompt to generate proposals; 2) Verb Prompt (VP): We provide patterns of nouns and the masked verb to build a prompt to generate pseudo queries with verbs. Our PZVMR utilizes task-relevant knowledge distilled from pre-trained CLIP and adapts the knowledge to VMR. Unlike the pioneering work, we introduce visual features into each module. Extensive experiments show that our PZVMR not only outperforms the existing zero-shot method (PSVL) on two public datasets (Charades-STA and ActivityNet-Captions) by 4.4% and 2.5% respectively in mIoU, but also outperforms several methods using stronger supervision.

## :fire: Updates

- [07/2024] Code of KPSC-P is released. Description of KPSC-F is released.