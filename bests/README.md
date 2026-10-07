# Zbot best checkpoints

- `zbot_walking_model_599.pt`: selected Walking checkpoint; metadata in `zbot_walking_model_599.json`.
- `zbot_disturbed_walking_model_599.pt`: selected disturbance-resistant Walking checkpoint from `2026-10-07_14-37-06`, iteration 599. Its saved training configurations are `disturbed_walking_env.yaml` and `disturbed_walking_agent.yaml`.

Play the disturbance-resistant model:

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Disturbed-Walking \
    --checkpoint-file bests/zbot_disturbed_walking_model_599.pt
```
