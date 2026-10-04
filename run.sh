source /home/yhzhu/AI/mjlab/.venv/bin/activate

# python play.py Mjlab-Zbot-6dof-Periodic-Stepping \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 16 \

# python play.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.2 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 16 \
#     # --checkpoint-file bests/zbot_walking_model_599.pt
#     # --checkpoint-file logs/rsl_rl/zbot_walking/2026-10-04_20-04-52/model_90.pt


python play.py Mjlab-Zbot-6dof-Step-Length-Walking \
    --env.step-frequency-min 0.2 \
    --env.step-frequency-max 1.0 \
    --env.scene.num-envs 16 \

# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.5 \
#     --env.step-frequency-max 0.5 \
#     --env.scene.num-envs 4 \
#     # --env.events.base-disturbance.params.force-range "(0.0,0.0)" \
#     # --env.events.base-disturbance.params.torque-range "(0.0,0.0)"