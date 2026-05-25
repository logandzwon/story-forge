# v3 1-step distill — checkpoint eval

Metric: per-frame LPIPS, student 1-step render vs teacher 4-step render (same still/prompt/seed). Eval set: first 11 prompts.
**Measured 1-step wall @ 192×192: 0.2056** (raw lightx2v 1-step vs 4-step teacher, same res; literature ref 1.09). Lower is better.

| checkpoint | mean LPIPS | beats wall? |
|---|---|---|
| student_step_0025.safetensors | 0.1291 | ✅ |
| student_step_0050.safetensors | 0.1178 | ✅ |
| student_step_0075.safetensors | 0.1098 | ✅ |
| student_step_0100.safetensors | 0.1151 | ✅ |
| student_step_0125.safetensors | 0.1145 | ✅ |
| student_step_0150.safetensors | 0.0823 | ✅ |

**Best: student_step_0150.safetensors @ LPIPS 0.0823** (beats the wall), vs measured wall 0.2056.
