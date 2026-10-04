source /home/yhzhu/AI/mjlab/.venv/bin/activate
python play.py Mjlab-Zbot-6dof-Walking \
    --env.step-frequency-min 0.25 \
    --env.step-frequency-max 1.0 \
    --env.scene.num-envs 16 \
    --checkpoint-file bests/zbot_walking_model_599.pt
