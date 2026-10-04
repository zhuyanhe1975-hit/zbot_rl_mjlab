source /home/yhzhu/AI/mjlab/.venv/bin/activate

# python train.py Mjlab-Zbot-6dof-Periodic-Stepping \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 100

# python train.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.2 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 600 \
#     --agent.save-interval 10

python train.py Mjlab-Zbot-6dof-Step-Length-Walking \
    --env.step-frequency-min 0.2 \
    --env.step-frequency-max 1.0 \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 300 \


# python train.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.5 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 200
