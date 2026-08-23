# (LLMLingua-2 + reranker) with vs without sentence dedup

- LLMLingua-2: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | reranker: `BAAI/bge-reranker-v2-m3` | MMR embedder: `BAAI/bge-small-en-v1.5`
- reader: `claude-haiku-4-5-20251001` @ temperature=0 | redundancy_threshold=0.8

## Task: ramble

| arm (rate \| dedup) | retention | passages kept | distractors leaked | mean F1 | EM | correct | single-mention |
|---|---|---|---|---|---|---|---|
| 0.75|L2 | 0.729 | (single doc) | - | 1.0 | 7/7 | 7/7 | 3/3 |
| 0.75|L2+MMR | 0.725 | (single doc) | - | 1.0 | 7/7 | 7/7 | 3/3 |
| 0.5|L2 | 0.515 | (single doc) | - | 1.0 | 7/7 | 7/7 | 3/3 |
| 0.5|L2+MMR | 0.508 | (single doc) | - | 1.0 | 7/7 | 7/7 | 3/3 |

<details><summary>ramble - 0.75|L2 per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Where is the speaker going camping? | Pinecrest Lake |  | Pinecrest Lake | 1.0 |  |
| How long is the drive? | three hours | Y | Around three hours. | 1.0 |  |
| Who is going with the speaker? | Dana |  | Dana | 1.0 |  |
| What day are they leaving? | Saturday | Y | Saturday morning. | 1.0 |  |
| What kind of fish do they hope to catch? | trout |  | Trout. | 1.0 |  |
| What did they forget last time? | the tent |  | Tent | 1.0 |  |
| When are they heading back? | Sunday night | Y | Sunday night. | 1.0 |  |

faults: none | MMR dropped sentences: 0

</details>


<details><summary>ramble - 0.75|L2+MMR per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Where is the speaker going camping? | Pinecrest Lake |  | Pinecrest Lake | 1.0 |  |
| How long is the drive? | three hours | Y | Around three hours. | 1.0 |  |
| Who is going with the speaker? | Dana |  | Dana | 1.0 |  |
| What day are they leaving? | Saturday | Y | Saturday morning. | 1.0 |  |
| What kind of fish do they hope to catch? | trout |  | Trout | 1.0 |  |
| What did they forget last time? | the tent |  | Tent | 1.0 |  |
| When are they heading back? | Sunday night | Y | Sunday night. | 1.0 |  |

faults: none | MMR dropped sentences: 1

</details>


<details><summary>ramble - 0.5|L2 per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Where is the speaker going camping? | Pinecrest Lake |  | Pinecrest Lake | 1.0 |  |
| How long is the drive? | three hours | Y | Around three hours. | 1.0 |  |
| Who is going with the speaker? | Dana |  | Dana | 1.0 |  |
| What day are they leaving? | Saturday | Y | Saturday morning | 1.0 |  |
| What kind of fish do they hope to catch? | trout |  | Trout | 1.0 |  |
| What did they forget last time? | the tent |  | Tent | 1.0 |  |
| When are they heading back? | Sunday night | Y | Sunday night | 1.0 |  |

faults: none | MMR dropped sentences: 0

</details>


<details><summary>ramble - 0.5|L2+MMR per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Where is the speaker going camping? | Pinecrest Lake |  | Pinecrest Lake | 1.0 |  |
| How long is the drive? | three hours | Y | Around three hours. | 1.0 |  |
| Who is going with the speaker? | Dana |  | Dana | 1.0 |  |
| What day are they leaving? | Saturday | Y | Saturday morning | 1.0 |  |
| What kind of fish do they hope to catch? | trout |  | Trout | 1.0 |  |
| What did they forget last time? | the tent |  | Tent | 1.0 |  |
| When are they heading back? | Sunday night | Y | Sunday night | 1.0 |  |

faults: none | MMR dropped sentences: 1

</details>

## Task: multidoc

| arm (rate \| dedup) | retention | passages kept | distractors leaked | mean F1 | EM | correct | single-mention |
|---|---|---|---|---|---|---|---|
| 0.75|L2 | 0.586 | 6/8 | 7 | 0.755 | 6/11 | 9/11 | 9/11 |
| 0.75|L2+MMR | 0.586 | 6/8 | 7 | 0.755 | 6/11 | 9/11 | 9/11 |
| 0.5|L2 | 0.419 | 6/8 | 7 | 0.745 | 6/11 | 9/11 | 9/11 |
| 0.5|L2+MMR | 0.419 | 6/8 | 7 | 0.745 | 6/11 | 9/11 | 9/11 |

<details><summary>multidoc - 0.75|L2 per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Who is the CEO of Nimbus? | Lena Ortiz | Y | Lena Ortiz | 1.0 |  |
| In what year was Nimbus founded? | 2021 | Y | 2021 | 1.0 |  |
| How much does the Pro plan cost per month? | $49 | Y | $49 | 1.0 |  |
| What is the overage cost per extra API call? | $0.002 | Y | $0.002 per additional call | 0.857 |  |
| How often does the data pipeline refresh? | every 15 minutes | Y | Every 15 minutes. | 1.0 |  |
| What uptime SLA does Nimbus promise? | 99.95% | Y | 99.95% | 1.0 |  |
| What database does Nimbus use as its primary store? | PostgreSQL | Y | PostgreSQL cluster | 0.667 |  |
| How long did the March 3 2024 outage last? | 47 minutes | Y | 47 minutes | 1.0 |  |
| What HTTP header authenticates API requests? | X-Nimbus-Key | Y | UNKNOWN | 0.0 | INFO_LOST_IN_COMPRESSION |
| What is the API rate limit? | 60 requests per minute | Y | Free tier: 1,000 calls/day; Pro: hig | 0.118 | INFO_LOST_IN_COMPRESSION |
| How many paid time off days do new employees get? | 20 days | Y | 20 days per year | 0.667 |  |

faults: {'INFO_LOST_IN_COMPRESSION': 2} | MMR dropped sentences: 0

</details>


<details><summary>multidoc - 0.75|L2+MMR per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Who is the CEO of Nimbus? | Lena Ortiz | Y | Lena Ortiz | 1.0 |  |
| In what year was Nimbus founded? | 2021 | Y | 2021 | 1.0 |  |
| How much does the Pro plan cost per month? | $49 | Y | $49 | 1.0 |  |
| What is the overage cost per extra API call? | $0.002 | Y | $0.002 per additional call | 0.857 |  |
| How often does the data pipeline refresh? | every 15 minutes | Y | Every 15 minutes. | 1.0 |  |
| What uptime SLA does Nimbus promise? | 99.95% | Y | 99.95% | 1.0 |  |
| What database does Nimbus use as its primary store? | PostgreSQL | Y | PostgreSQL cluster | 0.667 |  |
| How long did the March 3 2024 outage last? | 47 minutes | Y | 47 minutes | 1.0 |  |
| What HTTP header authenticates API requests? | X-Nimbus-Key | Y | UNKNOWN | 0.0 | INFO_LOST_IN_COMPRESSION |
| What is the API rate limit? | 60 requests per minute | Y | Free tier: 1,000 calls/day; Pro: hig | 0.118 | INFO_LOST_IN_COMPRESSION |
| How many paid time off days do new employees get? | 20 days | Y | 20 days per year | 0.667 |  |

faults: {'INFO_LOST_IN_COMPRESSION': 2} | MMR dropped sentences: 0

</details>


<details><summary>multidoc - 0.5|L2 per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Who is the CEO of Nimbus? | Lena Ortiz | Y | Lena Ortiz | 1.0 |  |
| In what year was Nimbus founded? | 2021 | Y | 2021 | 1.0 |  |
| How much does the Pro plan cost per month? | $49 | Y | $49 | 1.0 |  |
| What is the overage cost per extra API call? | $0.002 | Y | $0.002 per additional call | 0.857 |  |
| How often does the data pipeline refresh? | every 15 minutes | Y | Every 15 minutes. | 1.0 |  |
| What uptime SLA does Nimbus promise? | 99.95% | Y | 99.95% | 1.0 |  |
| What database does Nimbus use as its primary store? | PostgreSQL | Y | PostgreSQL cluster | 0.667 |  |
| How long did the March 3 2024 outage last? | 47 minutes | Y | 47 minutes | 1.0 |  |
| What HTTP header authenticates API requests? | X-Nimbus-Key | Y | UNKNOWN | 0.0 | INFO_LOST_IN_COMPRESSION |
| What is the API rate limit? | 60 requests per minute | Y | Free tier: 1,000 calls/day; Pro: hig | 0.0 | INFO_LOST_IN_COMPRESSION |
| How many paid time off days do new employees get? | 20 days | Y | 20 days per year | 0.667 |  |

faults: {'INFO_LOST_IN_COMPRESSION': 2} | MMR dropped sentences: 0

</details>


<details><summary>multidoc - 0.5|L2+MMR per-question</summary>

| Q | gold | single? | answer | F1 | fault |
|---|---|---|---|---|---|
| Who is the CEO of Nimbus? | Lena Ortiz | Y | Lena Ortiz | 1.0 |  |
| In what year was Nimbus founded? | 2021 | Y | 2021 | 1.0 |  |
| How much does the Pro plan cost per month? | $49 | Y | $49 | 1.0 |  |
| What is the overage cost per extra API call? | $0.002 | Y | $0.002 per additional call | 0.857 |  |
| How often does the data pipeline refresh? | every 15 minutes | Y | Every 15 minutes. | 1.0 |  |
| What uptime SLA does Nimbus promise? | 99.95% | Y | 99.95% | 1.0 |  |
| What database does Nimbus use as its primary store? | PostgreSQL | Y | PostgreSQL cluster | 0.667 |  |
| How long did the March 3 2024 outage last? | 47 minutes | Y | 47 minutes | 1.0 |  |
| What HTTP header authenticates API requests? | X-Nimbus-Key | Y | UNKNOWN | 0.0 | INFO_LOST_IN_COMPRESSION |
| What is the API rate limit? | 60 requests per minute | Y | Free tier: 1,000 calls/day; Pro: hig | 0.0 | INFO_LOST_IN_COMPRESSION |
| How many paid time off days do new employees get? | 20 days | Y | 20 days per year | 0.667 |  |

faults: {'INFO_LOST_IN_COMPRESSION': 2} | MMR dropped sentences: 0

</details>
