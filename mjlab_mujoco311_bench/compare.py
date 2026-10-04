"""Independent native MuJoCo 3.11 / MJLab comparison; no WorkBench code imports."""
import argparse,json,pathlib,sys,hashlib,subprocess
import numpy as np
import torch,mujoco
import mjlab
sys.path.insert(0,str(pathlib.Path.home()/'myWorks_vips/zbot_rl_mjlab/src'))
import zbot_rl_mjlab
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
ROOT=pathlib.Path(__file__).resolve().parent
class Actor(torch.nn.Module):
 def __init__(self,state):
  super().__init__(); self.register_buffer('mean',state['obs_normalizer._mean']); self.register_buffer('std',state['obs_normalizer._std'])
  self.net=torch.nn.Sequential(torch.nn.Linear(31,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,6))
  self.net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('mlp.')})
 def forward(self,x): return self.net((x-self.mean)/(self.std+1e-2))
def forward_axis(q):
 r=np.empty(9);mujoco.mju_quat2Mat(r,q);z=r.reshape(3,3)[:,2].copy();z[2]=0;z/=max(np.linalg.norm(z),1e-6);f=np.cross([0,0,-1],z);return f/max(np.linalg.norm(f),1e-6)
def observe(m,d,default,last,step,freq,initial):
 b=m.body('robot/base').id;r=d.xmat[b].reshape(3,3);vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,vel,0)
 # mj_objectVelocity uses the body COM; convert to the body link origin.
 lin=vel[3:]+np.cross(vel[:3],d.xpos[b]-d.xipos[b]);f=forward_axis(d.xquat[b]);yaw=np.arctan2(f[0]*initial[1]-f[1]*initial[0],np.dot(f[:2],initial[:2]))
 js=[m.joint(f'robot/joint{i}').id for i in range(1,7)]
 return np.concatenate([r.T@lin,r.T@vel[:3],r.T@np.array([0,0,-1]),[yaw],d.qpos[m.jnt_qposadr[js]]-default,d.qvel[m.jnt_dofadr[js]],last,[np.sin(step*.02*freq*2*np.pi),np.cos(step*.02*freq*2*np.pi)],[(freq-.2)/.8]]).astype(np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600);ap.add_argument('--device',default='cpu');ap.add_argument('--open-loop',action='store_true');ap.add_argument('--libtorch',action='store_true');args=ap.parse_args()
 cfg=load_env_cfg('Mjlab-Zbot-6dof-Walking',play=True);cfg.test_frequency=.4;cfg.scene.num_envs=1;cfg.terminations={};cfg.episode_length_s=1000
 env=ManagerBasedRlEnv(cfg,device=args.device);agent=load_rl_cfg('Mjlab-Zbot-6dof-Walking');wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 runner=MjlabOnPolicyRunner(wrapped,asdict(agent),device=args.device);runner.load(str(ROOT/'model.pt'),load_cfg={'actor':True},strict=True,map_location=args.device);reference=runner.get_inference_policy(device=args.device)
 obs=wrapped.reset()[0];robot=env.scene['robot'];m=env.sim._mj_model;d=mujoco.MjData(m)
 d.qpos[:]=env.sim.data.qpos[0].detach().cpu().numpy();d.qvel[:]=env.sim.data.qvel[0].detach().cpu().numpy();mujoco.mj_forward(m,d)
 default=robot.data.default_joint_pos[0].detach().cpu().numpy();actor=Actor(torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)['actor_state_dict']).eval();torch.jit.trace(actor,torch.zeros(1,31)).save(str(ROOT/'actor.ts'))
 cpp=subprocess.Popen([str(ROOT/'libtorch_smoke'),str(ROOT/'actor.ts')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) if args.libtorch else None
 def infer_native(o):
  if cpp is None:return actor(torch.tensor(o)[None]).numpy()[0]
  cpp.stdin.write(' '.join(str(float(v)) for v in o)+'\n');cpp.stdin.flush();return np.fromstring(cpp.stdout.readline(),sep=' ',dtype=np.float32)
 last=np.zeros(6);initial=forward_axis(d.xquat[m.body('robot/base').id]);trace=[];parity=0; synchronized_error=0
 with torch.inference_mode():
  for step in range(args.steps):
   refobs=obs['actor'][0].detach().cpu().numpy();refaction=reference(obs);parity=max(parity,float(np.max(np.abs(actor(torch.tensor(refobs)[None]).numpy()[0]-refaction[0].cpu().numpy()))))
   check=mujoco.MjData(m);check.qpos[:]=env.sim.data.qpos[0].cpu().numpy();check.qvel[:]=env.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,check)
   checkobs=observe(m,check,default,env.action_manager.action[0].cpu().numpy(),step,.4,initial)
   synchronized_error=max(synchronized_error,float(np.max(np.abs(checkobs-refobs))))
   nativeobs=observe(m,d,default,last,step,.4,initial);action=infer_native(nativeobs);last=action.copy();d.ctrl[:]=default+.25*(refaction[0].cpu().numpy() if args.open_loop else action)
   for _ in range(cfg.decimation):mujoco.mj_step(m,d)
   mujoco.mj_forward(m,d)
   obs=wrapped.step(refaction)[0]
   native_base=d.xpos[m.body('robot/base').id].copy();mj_base=robot.data.body_link_pos_w[0,list(robot.body_names).index('base')].detach().cpu().numpy();qerr=float(np.max(np.abs(d.qpos[:13]-env.sim.data.qpos[0].detach().cpu().numpy()[:13])))
   trace.append({'step':step+1,'observation_error':float(np.max(np.abs(refobs-nativeobs))),'action_error':float(np.max(np.abs(refaction[0].cpu().numpy()-action))),'base_error':float(np.linalg.norm(native_base-mj_base)),'qpos_error':qerr,'native_base':native_base.tolist(),'mjlab_base':mj_base.tolist()})
 first_div=next((x['step'] for x in trace if x['base_error']>.01),None)
 result={'mujoco_version':mujoco.__version__,'device':args.device,'steps':args.steps,'inference_parity_max':parity,'synchronized_observation_error_max':synchronized_error,'open_loop':args.open_loop,'libtorch':args.libtorch,'first_base_error_over_1cm':first_div,'max_base_error':max(x['base_error'] for x in trace),'trace':trace,'note':'Open-loop mode feeds exactly the MJLab-generated position actions to both engines. Auto-reset disabled. Native actor is also available through TorchScript/LibTorch.'}
 (ROOT/f'comparison-{args.device.replace(":","-")}{"-open" if args.open_loop else ""}{"-libtorch" if args.libtorch else ""}.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k!='trace'}));print('FIRST',trace[0]);print('LAST',trace[-1]);env.close()
 if cpp:cpp.stdin.close();cpp.wait(timeout=5)
if __name__=='__main__':main()
