source /home/yhzhu/AI/mjlab/.venv/bin/activate

# python play.py Mjlab-Zbot-6dof-Periodic-Stepping \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 16 \

python play.py Mjlab-Zbot-6dof-Walking \
    --env.step-frequency-min 0.2 \
    --env.step-frequency-max 0.2 \
    --env.scene.num-envs 64 \
    --checkpoint-file logs/rsl_rl/zbot_disturbed_walking/2026-10-04_15-45-55/model_499.pt


# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.2 \
#     --env.step-frequency-max 0.2 \
#     --env.scene.num-envs 64 \
#     --env.events.base-disturbance.params.force-range "(0.0,0.0)" \
#     --env.events.base-disturbance.params.torque-range "(0.0,0.0)"