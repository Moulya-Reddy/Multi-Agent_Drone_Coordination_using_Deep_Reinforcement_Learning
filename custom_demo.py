#!/usr/bin/env python3
"""UNIVERSAL DEMO - Works with ANY model format"""
import os, sys, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import argparse

DEVICE = "cpu"
GRID = 100.0
N_DRONES = 6
N_ZONES = 3
ZONE_CAP = 2

ZONES = np.array([[20., 20.], [50., 80.], [80., 20.]], dtype=np.float32)
OBSTACLES = [(np.array([35., 50.]), 7.),
             (np.array([65., 50.]), 7.),
             (np.array([50., 35.]), 5.)]


def detect_architecture(checkpoint):
    """Auto-detect obs_dim and n_actions"""
    if isinstance(checkpoint, dict):
        if 'actors' in checkpoint:
            actor_sd = checkpoint['actors'][0]
            obs_dim = actor_sd['net.0.weight'].shape[1]
            n_act = actor_sd['net.4.weight'].shape[0]
        elif 'q' in checkpoint:
            obs_dim = checkpoint['q']['net.0.weight'].shape[1]
            n_act = checkpoint['q']['net.4.weight'].shape[0]
        elif 'net.0.weight' in checkpoint:
            obs_dim = checkpoint['net.0.weight'].shape[1]
            n_act = checkpoint['net.4.weight'].shape[0]
        else:
            raise ValueError("Unknown format")
    elif isinstance(checkpoint, list):
        first = checkpoint[0]
        if 'sh.0.weight' in first:
            obs_dim = first['sh.0.weight'].shape[1]
            n_act = first['ac.weight'].shape[0]
        elif 'shared.0.weight' in first:
            obs_dim = first['shared.0.weight'].shape[1]
            n_act = first['actor.weight'].shape[0]
        elif 'net.0.weight' in first:
            obs_dim = first['net.0.weight'].shape[1]
            n_act = first['net.4.weight'].shape[0]
        else:
            raise ValueError("Unknown list format")
    else:
        raise ValueError("Unknown type")
    return obs_dim, n_act


class DQNAgent:
    class Net(nn.Module):
        def __init__(self, obs_dim, n_act):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),
                                      nn.Linear(128,128),nn.ReLU(),
                                      nn.Linear(128,n_act))
        def forward(self,x): return self.net(x)

    def __init__(self, obs_dim, n_act):
        self.q = self.Net(obs_dim, n_act).to(DEVICE)

    def load(self, checkpoint):
        if isinstance(checkpoint, dict):
            if 'q' in checkpoint: self.q.load_state_dict(checkpoint['q'])
            else: self.q.load_state_dict(checkpoint)
        else: self.q.load_state_dict(checkpoint)

    def act(self, obs_dict):
        acts = {}
        for ag, o in obs_dict.items():
            with torch.no_grad():
                acts[ag] = self.q(torch.FloatTensor(o).unsqueeze(0).to(DEVICE)).argmax().item()
        return acts


class PPOAgent:
    class AC(nn.Module):
        def __init__(self, obs_dim, n_act):
            super().__init__()
            self.sh=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.ac=nn.Linear(128,n_act); self.cr=nn.Linear(128,1)
        def forward(self,x): h=self.sh(x); return self.ac(h),self.cr(h)
    
    class ACOld(nn.Module):
        def __init__(self, obs_dim, n_act):
            super().__init__()
            self.shared=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.actor=nn.Linear(128,n_act); self.critic=nn.Linear(128,1)
        def forward(self,x): h=self.shared(x); return self.actor(h),self.critic(h)

    def __init__(self, obs_dim, n_act, old_naming=False):
        if old_naming:
            self.nets = nn.ModuleList([self.ACOld(obs_dim,n_act).to(DEVICE) for _ in range(N_DRONES)])
        else:
            self.nets = nn.ModuleList([self.AC(obs_dim,n_act).to(DEVICE) for _ in range(N_DRONES)])

    def load(self, checkpoint):
        for net, sd in zip(self.nets, checkpoint): net.load_state_dict(sd)

    def act(self, obs_dict):
        acts = {}
        for i, ag in enumerate([f"drone_{j}" for j in range(N_DRONES)]):
            with torch.no_grad():
                s = torch.FloatTensor(obs_dict[ag]).unsqueeze(0).to(DEVICE)
                acts[ag] = self.nets[i](s)[0].argmax().item()
        return acts


class MADDPGAgent:
    class Actor(nn.Module):
        def __init__(self, obs_dim, n_act):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),nn.Linear(128,n_act))
        def forward(self,x): return self.net(x)

    def __init__(self, obs_dim, n_act):
        self.actors = nn.ModuleList([self.Actor(obs_dim,n_act).to(DEVICE) for _ in range(N_DRONES)])

    def load(self, checkpoint):
        actor_states = checkpoint['actors'] if isinstance(checkpoint,dict) and 'actors' in checkpoint else checkpoint
        for i, actor in enumerate(self.actors): actor.load_state_dict(actor_states[i])

    def act(self, obs_dict):
        acts = {}
        for i in range(N_DRONES):
            ag = f"drone_{i}"
            with torch.no_grad():
                acts[ag] = self.actors[i](torch.FloatTensor(obs_dict[ag]).unsqueeze(0).to(DEVICE)).argmax().item()
        return acts


class Env:
    def __init__(self, obs_dim, n_act):
        self.obs_dim, self.n_act = obs_dim, n_act
        self.agents = [f"drone_{i}" for i in range(N_DRONES)]
        self.dirs = np.array([[np.cos(k*2*np.pi/n_act), np.sin(k*2*np.pi/n_act)]
                              for k in range(n_act)], dtype=np.float32)

    def reset_custom(self, positions, targets):
        self.pos = np.array(positions, dtype=np.float32)
        self.vel = np.zeros((N_DRONES, 2), dtype=np.float32)
        self.target = np.array(targets, dtype=int)
        self.done = np.zeros(N_DRONES, dtype=bool)
        self.t = 0
        self.coll_dd = self.coll_do = 0
        self.trajectory = [self.pos.copy()]
        
        print(f"\n📍 Scenario (obs_dim={self.obs_dim}, actions={self.n_act}):")
        for i in range(N_DRONES):
            print(f"  D{i}: {self.pos[i]} → Z{self.target[i]}")
        print()
        return self._obs()

    def _obs(self):
        out = {}
        zc = np.zeros(N_ZONES, dtype=float)
        for i in range(N_DRONES):
            if not self.done[i]: zc[self.target[i]] += 1.
        
        for i, ag in enumerate(self.agents):
            tgt = ZONES[self.target[i]]
            diff = tgt - self.pos[i]
            dist = np.linalg.norm(diff)
            tdir = diff / (dist + 1e-6)
            own = np.array([self.pos[i,0]/GRID, self.pos[i,1]/GRID,
                            self.vel[i,0]/4., self.vel[i,1]/4., tdir[0], tdir[1]], dtype=np.float32)
            
            oth = []
            for j in range(N_DRONES):
                if j == i: continue
                rp = (self.pos[j]-self.pos[i])/GRID
                rv = (self.vel[j]-self.vel[i])/4.
                st = float(self.target[j] == self.target[i])
                if self.obs_dim >= 40: oth += [rp[0], rp[1], rv[0], rv[1], st]
                else: oth += [rp[0], rp[1], rv[0], rv[1]]
            
            pads = []
            for k in range(N_ZONES):
                rp = (ZONES[k]-self.pos[i])/GRID
                d = np.linalg.norm(ZONES[k]-self.pos[i])/GRID
                if self.obs_dim >= 40:
                    conf = float(zc[k] > ZONE_CAP)
                    pads += [rp[0], rp[1], d, conf]
                else: pads += [rp[0], rp[1]]
            
            obs = np.concatenate([own, np.array(oth,dtype=np.float32), np.array(pads,dtype=np.float32)])
            if len(obs) < self.obs_dim: obs = np.pad(obs, (0, self.obs_dim-len(obs)))
            elif len(obs) > self.obs_dim: obs = obs[:self.obs_dim]
            out[ag] = obs.astype(np.float32)
        return out

    def step(self, acts):
        self.t += 1
        for i,ag in enumerate(self.agents):
            if self.done[i]: continue
            acc = self.dirs[acts[ag]] * 0.6
            self.vel[i] = self.vel[i]*0.90 + acc
            spd = np.linalg.norm(self.vel[i])
            if spd > 4.0: self.vel[i] = self.vel[i]/spd*4.0
            npos = np.clip(self.pos[i]+self.vel[i], 0., GRID)
            for o,r in OBSTACLES:
                if np.linalg.norm(npos-o) < r+3.5:
                    diff = npos-o
                    npos = o + diff/(np.linalg.norm(diff)+1e-6)*(r+4.5)
                    self.vel[i] *= 0.; self.coll_do += 1
            self.pos[i] = npos
        
        self.trajectory.append(self.pos.copy())
        for i in range(N_DRONES):
            if self.done[i]: continue
            z = self.target[i]
            if np.linalg.norm(self.pos[i]-ZONES[z]) < 8.:
                at = [j for j in range(N_DRONES) if not self.done[j] and np.linalg.norm(self.pos[j]-ZONES[z]) < 8.]
                if len(at) <= ZONE_CAP: self.done[i] = True
        
        for i in range(N_DRONES):
            if self.done[i]: continue
            for j in range(i+1, N_DRONES):
                if self.done[j]: continue
                if np.linalg.norm(self.pos[i]-self.pos[j]) < 5.: self.coll_dd += 1
        return self._obs()


def run_demo(env, agent, algo, max_steps=500):
    plt.ion()
    fig, ax = plt.subplots(figsize=(11,11), facecolor='white')
    for step in range(max_steps):
        obs = env._obs()
        acts = agent.act(obs)
        env.step(acts)
        if step % 5 == 0 or all(env.done):
            draw(ax, env, algo, step)
            plt.pause(0.1)
        if all(env.done):
            print(f"\n✅ Done in {step} steps!")
            plt.pause(3); break
    plt.ioff(); plt.show()
    return {'delivered':int(env.done.sum()), 'steps':step, 'collisions':env.coll_dd}


def draw(ax, env, algo, step):
    ax.clear()
    ax.set_xlim(-5,105); ax.set_ylim(-5,105)
    ax.set_facecolor('#e8f4f8'); ax.set_aspect('equal')
    ax.grid(True,alpha=0.3,lw=0.8,color='#cfd8dc',zorder=0)
    ax.set_xlabel("X Position (meters)",fontsize=11)
    ax.set_ylabel("Y Position (meters)",fontsize=11)
    for sp in ax.spines.values(): sp.set_color('#90a4ae'); sp.set_linewidth(1.5)
    
    nd=int(env.done.sum()); pct=int(nd*100/N_DRONES)
    ax.text(50,108,f"Multi-Agent Drone Fleet - {algo}\nStep: {step} | Delivered: {nd}/{N_DRONES} ({pct}%) | Collisions: {env.coll_dd}",
            ha='center',va='top',fontsize=11,fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8',facecolor='#e3f2fd',edgecolor='#2196f3',lw=2.5),zorder=100)
    
    lx,ly=88,100
    ax.scatter(lx,ly,marker='*',s=200,color='#ff9800',edgecolor='#ef6c00',lw=2,zorder=100)
    ax.text(lx+3,ly,'Target',fontsize=9,va='center',zorder=100)
    ax.scatter(lx,ly-6,marker='o',s=150,color='#1976d2',edgecolor='black',lw=1.5,zorder=100)
    ax.text(lx+3,ly-6,'Drone',fontsize=9,va='center',zorder=100)
    
    for o,r in OBSTACLES:
        ax.add_patch(Rectangle((o[0]-r/2,o[1]-r/2),r,r,facecolor='#546e7a',edgecolor='#37474f',lw=2.5,zorder=10))
        wsz=r/6.5; wsp=r/3.5
        for wx,wy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
            ax.add_patch(Rectangle((o[0]+wx*wsp-wsz/2,o[1]+wy*wsp-wsz/2),wsz,wsz,
                                   facecolor='#fdd835',edgecolor='#f9a825',lw=1,zorder=11))
        ax.add_patch(Polygon([[o[0]-r/2-0.3,o[1]+r/2],[o[0],o[1]+r/2+r*0.35],[o[0]+r/2+0.3,o[1]+r/2]],
                             facecolor='#607d8b',edgecolor='#455a64',lw=2.2,zorder=12))
    
    for k,z in enumerate(ZONES):
        for rad,alp in [(11,.07),(8,.11),(5.5,.16)]:
            ax.add_patch(Circle(z,rad,facecolor='#ffe082',alpha=alp,zorder=5))
        ax.scatter(z[0],z[1],marker='*',s=750,color='#ff9800',edgecolor='#ef6c00',lw=2.5,zorder=20)
        ax.text(z[0],z[1]-6,f"T{k}",color='#ef6c00',fontsize=8,ha='center',fontweight='bold',zorder=21,
                bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor='#ff9800',lw=1.5,alpha=0.9))
    
    COLS=['#d32f2f','#1976d2','#388e3c','#7b1fa2','#f57c00','#00acc1']
    NAMES=['Red','Blue','Green','Purple','Orange','Cyan']
    for i in range(N_DRONES):
        c=COLS[i]; traj=np.array([t[i] for t in env.trajectory])
        if len(traj)>2: ax.plot(traj[:,0],traj[:,1],color=c,lw=2,alpha=0.4,zorder=15)
        p=env.pos[i]
        if not env.done[i]:
            ax.add_patch(Circle((p[0]+.25,p[1]-.25),2.6,color='black',alpha=0.15,zorder=18))
            ax.add_patch(Circle(p,2.4,color=c,edgecolor='black',lw=2.5,alpha=0.95,zorder=25))
            ax.add_patch(Circle(p,1.2,color='white',edgecolor=c,lw=1.5,alpha=0.92,zorder=26))
            ax.add_patch(Circle(p,.55,color='black',alpha=0.9,zorder=27))
            for ang in [45,135,225,315]:
                rd=np.radians(ang)
                ax.add_patch(Circle((p[0]+np.cos(rd)*2.6,p[1]+np.sin(rd)*2.6),.85,
                                    facecolor='#424242',edgecolor='black',lw=1.5,zorder=28))
            ax.text(p[0],p[1]-4.8,f"{NAMES[i]} {i}",color='white',fontsize=9.5,ha='center',fontweight='bold',zorder=35,
                   bbox=dict(boxstyle='round,pad=0.35',facecolor=c,edgecolor='black',lw=2,alpha=0.95))
        else:
            ax.text(ZONES[env.target[i],0],ZONES[env.target[i],1]+3.5,'✓',
                   color='#2e7d32',fontsize=20,ha='center',fontweight='bold',zorder=30)


SCENARIOS = {
    "easy":    {"name":"Easy",     "pos":[[25,25],[30,30],[35,25],[55,75],[60,80],[65,75]], "tgt":[0,0,1,1,2,2]},
    "hard":    {"name":"Hard",     "pos":[[15,50],[85,50],[50,15],[50,85],[25,25],[75,75]], "tgt":[2,0,1,1,2,0]},
    "conflict":{"name":"Conflict", "pos":[[15,15],[85,15],[15,85],[85,85],[30,50],[70,50]], "tgt":[1,1,1,1,1,1]},
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--algo",required=True,choices=["DQN","PPO","MADDPG"])
    parser.add_argument("--scenario",default="hard",choices=list(SCENARIOS.keys())+["custom"])
    parser.add_argument("--positions",type=str)
    parser.add_argument("--targets",type=str)
    args=parser.parse_args()
    
    print("\n"+"="*60)
    print(f"  UNIVERSAL DEMO - {args.algo}")
    print("="*60+"\n")
    
    checkpoint=torch.load(args.model,map_location=DEVICE)
    obs_dim,n_act=detect_architecture(checkpoint)
    print(f"✅ Detected: obs_dim={obs_dim}, actions={n_act}")
    
    if args.algo=="DQN":
        agent=DQNAgent(obs_dim,n_act); agent.load(checkpoint)
    elif args.algo=="PPO":
        old_naming='shared.0.weight' in checkpoint[0]
        agent=PPOAgent(obs_dim,n_act,old_naming); agent.load(checkpoint)
    else:
        agent=MADDPGAgent(obs_dim,n_act); agent.load(checkpoint)
    print(f"✅ Loaded {args.algo}")
    
    env=Env(obs_dim,n_act)
    if args.scenario=="custom":
        if not args.positions or not args.targets:
            print("❌ Need --positions and --targets"); return
        pos=[[float(x) for x in p.split(',')] for p in args.positions.split()]
        tgt=[int(t) for t in args.targets.split()]
    else:
        s=SCENARIOS[args.scenario]
        pos,tgt=s["pos"],s["tgt"]
        print(f"📍 {s['name']}")
    
    env.reset_custom(pos,tgt)
    print("🎬 Starting...")
    result=run_demo(env,agent,args.algo,500)
    
    print("\n"+"="*60)
    print(f"  Delivered: {result['delivered']}/{N_DRONES}")
    print(f"  Steps:     {result['steps']}")
    print(f"  Collisions:{result['collisions']}")
    print("="*60+"\n")

if __name__=="__main__":
    main()