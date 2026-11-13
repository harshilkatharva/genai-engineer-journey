### Approach C (Grounded Reflection) Validation Summary (15 Topics)

| Topic | Rounds (Best) | R0 → Best Score (Delta) | Blinded Score | R0 → Best Defects (C/M) | R0 → Best Grounding | Status | Termination | LLM Calls | Attempts (Retries) | Tokens | Cost | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|
| `qc-01` | 3 (`R2`) | 0.8675 → 0.9925 (+0.1250) | 0.9925 | 0/7 → 0/1 | 0.7500 → 1.0000 | UNRESOLVED (`accepted=False`) | `CONSECUTIVE_REGRESSIONS_EXCEEDED (1 consecutive regressions)` | 40 | 43 (3) | 55,686 | $0.0078 | 418.3s |
| `qc-02` | 1 (`R1`) | 0.8875 → 0.9925 (+0.1050) | 0.9850 | 2/4 → 0/0 | 0.7222 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 17 | 23 (6) | 27,669 | $0.0040 | 239.0s |
| `qc-03` | 3 (`R3`) | 0.8075 → 0.9925 (+0.1850) | 0.9925 | 0/16 → 0/0 | 0.0000 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 40 | 44 (4) | 52,367 | $0.0073 | 608.3s |
| `ai-01` | 2 (`R2`) | 0.8025 → 0.9925 (+0.1900) | 0.9925 | 0/9 → 0/0 | 0.5714 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 23 | 40 (17) | 36,229 | $0.0051 | 754.1s |
| `ai-02` | 2 (`R2`) | 0.9100 → 0.9925 (+0.0825) | 0.9850 | 0/2 → 0/0 | 0.8889 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 31 | 37 (6) | 42,439 | $0.0059 | 339.8s |
| `ai-03` | 2 (`R2`) | 0.9475 → 0.9925 (+0.0450) | 0.9925 | 0/3 → 0/0 | 1.0000 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 27 | 30 (3) | 40,401 | $0.0057 | 261.2s |
| `energy-01` | 0 (`R0`) | 0.9850 → 0.9850 (+0.0000) | 0.9850 | 0/0 → 0/0 | 1.0000 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 9 | 13 (4) | 10,960 | $0.0016 | 100.3s |
| `energy-02` | 3 (`R3`) | 0.8375 → 0.9850 (+0.1475) | 0.9850 | 0/9 → 0/0 | 0.5000 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 31 | 37 (6) | 50,882 | $0.0070 | 662.2s |
| `energy-03` | 3 (`R3`) | 0.8550 → 1.0000 (+0.1450) | 1.0000 | 0/9 → 0/0 | 0.6667 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 39 | 42 (3) | 52,072 | $0.0071 | 354.4s |
| `bio-01` | 1 (`R1`) | 0.8225 → 0.9675 (+0.1450) | 0.9675 | 0/12 → 0/0 | 0.5000 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 15 | 20 (5) | 24,737 | $0.0035 | 372.7s |
| `bio-02` | 3 (`R3`) | 0.8150 → 0.9850 (+0.1700) | 0.9850 | 0/8 → 0/0 | 0.6250 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 31 | 36 (5) | 53,076 | $0.0074 | 207.1s |
| `bio-03` | 1 (`R0`) | 0.9850 → 0.9850 (+0.0000) | 0.9850 | 0/1 → 0/1 | 1.0000 → 1.0000 | UNRESOLVED (`accepted=False`) | `CONSECUTIVE_REGRESSIONS_EXCEEDED (1 consecutive regressions)` | 17 | 17 (0) | 21,804 | $0.0032 | 216.0s |
| `dist-01` | 2 (`R1`) | 0.8200 → 0.9850 (+0.1650) | 0.9850 | 0/13 → 0/3 | 0.4000 → 1.0000 | UNRESOLVED (`accepted=False`) | `CONSECUTIVE_REGRESSIONS_EXCEEDED (1 consecutive regressions)` | 23 | 34 (11) | 31,812 | $0.0043 | 831.5s |
| `dist-02` | 1 (`R1`) | 0.9925 → 0.9925 (+0.0000) | 0.9925 | 0/4 → 0/0 | 0.9375 → 1.0000 | ACCEPTED (`accepted=True`) | `ALL_ACCEPTANCE_CRITERIA_MET` | 21 | 24 (3) | 30,635 | $0.0045 | 105.1s |
| `dist-03` | 3 (`R3`) | 0.8800 → 0.9925 (+0.1125) | 0.9925 | 0/7 → 0/1 | 0.7500 → 0.9545 | UNRESOLVED (`accepted=False`) | `BUDGET_EXCEEDED: TOKENS_EXHAUSTED (used 55532 >= max 50000)` | 41 | 51 (10) | 55,532 | $0.0077 | 300.0s |

**Totals (15 topics):** Accepted `11/15` (`73.3%`) | LLM Calls: `405` | API Attempts (Retries): `491` (`86`) | Tokens: `586,301` | Cost: `$0.0820` | Sum Latency: `5769.9s`
