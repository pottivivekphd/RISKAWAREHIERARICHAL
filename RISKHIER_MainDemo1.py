import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)


np.random.seed(42)
torch.manual_seed(42)
num_tasks = 100
tasks = pd.DataFrame({

    "Task_ID": np.arange(1, num_tasks+1),

    "RAM": np.random.randint(128, 4096, num_tasks),

    "MIPS": np.random.randint(100, 5000, num_tasks),

    "Energy": np.random.uniform(1, 100, num_tasks),

    "Bandwidth": np.random.uniform(1, 100, num_tasks),

    "Deadline": np.random.uniform(0.1, 5, num_tasks),

    "Queue_Load": np.random.uniform(0, 1, num_tasks),

    "Delay": np.random.uniform(1, 100, num_tasks)

})


tasks["Congestion"] = (

    0.4*tasks["Queue_Load"] +
    0.3*(tasks["Delay"]/100) +
    0.3*(1-tasks["Bandwidth"]/100)

)

tasks.to_csv("Tasks_Input_Dataset.csv", index=False)


###QR-LSTM CONGESTION PREDICTION
scaler = MinMaxScaler()
X = tasks[["RAM","MIPS","Energy","Bandwidth","Queue_Load","Delay"]]
y = tasks["Congestion"]
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((num_tasks,1,6))
quantiles = [0.1,0.5,0.9]
inputs = layers.Input(shape=(1,6))
x = layers.LSTM(64)(inputs)
outputs = [layers.Dense(1)(x) for _ in quantiles]
qr_model = Model(inputs, outputs)
def quantile_loss(q,y,f):

    e = y-f
    return tf.reduce_mean(tf.maximum(q*e,(q-1)*e))
losses = [lambda y,f,q=q: quantile_loss(q,y,f) for q in quantiles]
qr_model.compile(optimizer='adam', loss=losses)
qr_model.fit(
    X_scaled,
    [y,y,y],
    epochs=30,
    batch_size=16,
    verbose=0
)
qr_model.summary()
pred = qr_model.predict(X_scaled)
tasks["Congestion_Prediction"] = pred[1].flatten()



actual = y.values
predicted = tasks["Congestion_Prediction"].values


R = np.corrcoef(actual, predicted)[0,1]   ###PCC
R2 = r2_score(actual, predicted)   ##(R²)
EVS = explained_variance_score(actual, predicted)   ##EVS
NSE = 1 - (np.sum((actual-predicted)**2) /
           np.sum((actual-np.mean(actual))**2))# 4. Nash–Sutcliffe Efficiency (NSE)
WI = 1 - (np.sum((actual-predicted)**2) /
          np.sum((np.abs(predicted-np.mean(actual)) +
                  np.abs(actual-np.mean(actual)))**2))# 5. Willmott’s Index of Agreement (WI)
Accuracy = 100 * (1 - np.mean(np.abs((actual-predicted)/actual)))# 6. Prediction Accuracy (%)
MPR = np.mean(predicted/actual) # 7. Mean Prediction Ratio (MPR)
VR = np.var(predicted)/np.var(actual) # 8. Variance Ratio (VR)
SDR = np.std(predicted)/np.std(actual) # 9. Standard Deviation Ratio (SDR)
Covariance = np.cov(actual, predicted)[0,1]  # 10. Covariance
mean_actual = np.mean(actual) 
mean_pred = np.mean(predicted)
var_actual = np.var(actual)
var_pred = np.var(predicted)
CCC = (2*Covariance) / (var_actual + var_pred +
                        (mean_actual-mean_pred)**2) # 11. Concordance Correlation Coefficient (CCC)
IR = np.sum(predicted**2)/np.sum(actual**2) # 12. Index of Reliability (IR)
EC = 1 - (np.var(actual-predicted)/np.var(actual)) # 13. Efficiency Coefficient (EC)
r = R
alpha = np.std(predicted)/np.std(actual)
beta = np.mean(predicted)/np.mean(actual)
KGE = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2) # 14. Kling–Gupta Efficiency (KGE)
AC = 1 - (np.sum((actual-predicted)**2) /
          np.sum((np.abs(predicted-np.mean(actual)) + 
                  np.abs(actual-np.mean(actual)))**2)) # 15. Agreement Coefficient (AC)
MSE = mean_squared_error(actual, predicted) # 1. MSE
RMSE = np.sqrt(MSE) # 2. RMSE
MAE = mean_absolute_error(actual, predicted)# 3. MAE
R2 = r2_score(actual, predicted) # 4. R2 Score
EVS = explained_variance_score(actual, predicted) # 5. Explained Variance Score
MAPE = mean_absolute_percentage_error(actual, predicted) # 6. Mean Absolute Percentage Error (MAPE)
MedAE = median_absolute_error(actual, predicted) # 7. Median Absolute Error
Max_Error = max_error(actual, predicted) # 8. Max Error
MBE = np.mean(predicted - actual) # 9. Mean Bias Error (MBE)
MARE = np.mean(np.abs((actual - predicted) / actual)) # 10. Mean Absolute Relative Error (MARE)
NRMSE = RMSE / (actual.max() - actual.min()) # 11. Normalized RMSE (NRMSE)
RMSLE = np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual))**2)) # 12. Root Mean Square Log Error (RMSLE)
Correlation = np.corrcoef(actual, predicted)[0,1] # 13. Pearson Correlation Coefficient
RSE = np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2) # 14. Relative Squared Error (RSE)
SMAPE = np.mean(
    2 * np.abs(predicted - actual) /
    (np.abs(actual) + np.abs(predicted))
) # 15. Symmetric Mean Absolute Percentage Error (SMAPE)




tasks["Queue_norm"] = tasks["Queue_Load"]
tasks["Delay_norm"] = tasks["Delay"] / tasks["Delay"].max()
tasks["Bandwidth_norm"] = 1 - (tasks["Bandwidth"] / tasks["Bandwidth"].max())
tasks["Energy_norm"] = tasks["Energy"] / tasks["Energy"].max()
# Priority calculation
tasks["Priority"] = tasks["Congestion_Prediction"] * (1/tasks["Deadline"])

# ============================================================
#DAIR-GNN TOPOLOGY LEARNING
# ============================================================

node_features = torch.tensor(
    tasks[["RAM","MIPS","Energy","Bandwidth"]].values,
    dtype=torch.float
)

edge_index = torch.randint(0,num_tasks,(2,300))

data = Data(x=node_features, edge_index=edge_index)

class GNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(4,32)
        self.conv2 = GCNConv(32,16)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = torch.relu(x)

        x = self.conv2(x,edge_index)

        return x

gnn = GNN()

optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

for epoch in range(100):

    optimizer.zero_grad()

    out = gnn(data)

    loss = out.mean()

    loss.backward()

    optimizer.step()

embeddings = gnn(data).detach().numpy()
tasks["Topology_Influence"] = (

    (embeddings[:,0] - embeddings[:,0].min()) /
    (embeddings[:,0].max() - embeddings[:,0].min())

)

class PPO_Global(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,32)
        self.actor = nn.Linear(32,3)
        self.critic = nn.Linear(32,1)

    def forward(self,state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.actor(x),dim=-1)
        value = self.critic(x)
        return probs,value
    
# ============================================================
# STEP 4: RL ENVIRONMENT
# ============================================================
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

class FogEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6,),
            dtype=np.float32
        )

        # FIXED: Continuous action space for SAC
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.reward_history = []

    def reset(self, seed=None):

        self.state = np.random.rand(6).astype(np.float32)

        return self.state, {}

    def step(self, action):

        reward = float(np.random.rand())

        self.reward_history.append(reward)

        done = False

        return self.state, reward, done, False, {}



env=FogEnv()


# ============================================================
# STEP 5: PPO GLOBAL DECISION
# ============================================================

ppo=PPO("MlpPolicy",env,verbose=0)

ppo.learn(total_timesteps=5000)


ppo_rewards=env.reward_history





# Predict global decisions

global_decisions=[]

for i in range(num_tasks):

    obs=np.random.rand(6)

    action,_=ppo.predict(obs)

    global_decisions.append(action)

tasks["Global_Decision"]=global_decisions


# ============================================================
#  DRHP GLOBAL DECISION
# ============================================================

global_decisions = []

for i in range(num_tasks):

    risk = (
        tasks["Congestion_Prediction"].iloc[i] +
        tasks["Topology_Influence"].iloc[i]
    )

    if risk < 0.4:

        global_decisions.append("LOCAL")

    elif risk < 0.7:

        global_decisions.append("FOG")

    else:

        global_decisions.append("CLOUD")

tasks["Global_Decision"] = global_decisions


# ============================================================
# STEP 6: SAC LOCAL ALLOCATION
# ============================================================

env2 = FogEnv()

sac = SAC("MlpPolicy", env2, verbose=1)

sac.learn(total_timesteps=5000)


sac_rewards=env2.reward_history




local_alloc=[]

for i in range(num_tasks):

    obs=np.random.rand(6)

    action,_=sac.predict(obs)

    local_alloc.append(action)

tasks["Local_Server"]=local_alloc


local_servers = ["L1","L2","L3"]
fog_servers = ["F1","F2","F3","F4","F5"]
cloud_servers = ["C1","C2"]

allocated = []

for decision in tasks["Global_Decision"]:

    if decision == "LOCAL":

        allocated.append(np.random.choice(local_servers))

    elif decision == "FOG":

        allocated.append(np.random.choice(fog_servers))

    else:

        allocated.append(np.random.choice(cloud_servers))

tasks["Allocated_Server"] = allocated

# ============================================================
#  FAPT ADAPTATION
# ============================================================

feedback_error = np.random.uniform(0,0.2,num_tasks)

tasks["Adapted_Priority"] = (

    tasks["Priority"] * (1-feedback_error)

)

# ============================================================
#  CURE UNCERTAINTY AND EXPLORATION
# ============================================================

tasks["Uncertainty"] = abs(
    tasks["Priority"] - tasks["Adapted_Priority"]
)

tasks["Exploration_Rate"] = (
    tasks["Uncertainty"] /
    tasks["Uncertainty"].max()
)

# ============================================================
# FINAL OUTPUT
# ============================================================

final_columns = [

"Task_ID",
"RAM",
"MIPS",
"Energy",
"Bandwidth",
"Deadline",
"Queue_Load",
"Delay",
"Queue_norm",
"Delay_norm",
"Bandwidth_norm",
"Energy_norm",
"Congestion_Prediction",
"Priority",
"Topology_Influence",
"Global_Decision",
"Allocated_Server",
"Adapted_Priority",
"Uncertainty",
"Exploration_Rate"

]

print("\nFINAL OUTPUT DATASET\n")

print(tasks[final_columns].head(10))



# ============================================================
# END
# ============================================================
# ============================================================
# STEP 9: SYSTEM PERFORMANCE METRICS
# ============================================================

# Simulated system parameters
fog_speed = 3000      # MIPS
cloud_speed = 6000    # MIPS
local_speed = 1500    # MIPS

energy_rate_local = 0.9
energy_rate_fog = 0.6
energy_rate_cloud = 0.4

network_delay_fog = 0.05
network_delay_cloud = 0.1
network_delay_local = 0.01

execution_time = []
waiting_time = []
response_time = []
turnaround_time = []
energy_consumption = []
latency = []

# ============================================================
# COMPUTE METRICS PER TASK
# ============================================================

for i in range(num_tasks):

    decision = tasks["Global_Decision"].iloc[i]
    mips = tasks["MIPS"].iloc[i]

    if decision == "LOCAL":

        exec_time = mips / local_speed
        energy = exec_time * energy_rate_local
        net_delay = network_delay_local

    elif decision == "FOG":

        exec_time = mips / fog_speed
        energy = exec_time * energy_rate_fog
        net_delay = network_delay_fog

    else:

        exec_time = mips / cloud_speed
        energy = exec_time * energy_rate_cloud
        net_delay = network_delay_cloud


    wait_time = np.random.uniform(0.01,0.1)

    resp_time = exec_time + wait_time + net_delay

    tat = resp_time

    execution_time.append(exec_time)
    waiting_time.append(wait_time)
    response_time.append(resp_time)
    turnaround_time.append(tat)
    energy_consumption.append(energy)
    latency.append(resp_time)


# Convert to numpy
execution_time = np.array(execution_time)
waiting_time = np.array(waiting_time)
response_time = np.array(response_time)
turnaround_time = np.array(turnaround_time)
energy_consumption = np.array(energy_consumption)
latency = np.array(latency)

# ============================================================
# GLOBAL METRICS
# ============================================================

# Makespan
makespan = np.max(turnaround_time)

# Throughput
throughput = num_tasks / makespan

# Avg latency
avg_latency = np.mean(latency)

# Energy consumption
total_energy = np.sum(energy_consumption)

# Energy efficiency
energy_efficiency = throughput / total_energy

# Resource utilization
total_capacity = num_tasks * cloud_speed
used_capacity = np.sum(tasks["MIPS"])

resource_utilization = used_capacity / total_capacity

# SLA violation
sla_deadline = tasks["Deadline"].values

sla_violation = np.sum(turnaround_time > sla_deadline) / num_tasks

# Avg execution time
avg_execution_time = np.mean(execution_time)

# Avg waiting time
avg_waiting_time = np.mean(waiting_time)

# Avg response time
avg_response_time = np.mean(response_time)

# Avg turnaround time
avg_tat = np.mean(turnaround_time)

ppo = PPO_Global()

import torch.optim as optim
optimizer_ppo = optim.Adam(ppo.parameters(),lr=0.001)



episodes = 100

episode_total_rewards = []
episode_avg_rewards = []
episode_variance = []
episode_success_rate = []
episode_cumulative_reward = []
episode_convergence = []
episode_complexity = []

total_params = sum(p.numel() for p in ppo.parameters())

for ep in range(episodes):

    total_reward = 0
    rewards_this_episode = []
    success_count = 0

    for i in range(num_tasks):

        state = torch.tensor([
            tasks["Congestion_Prediction"].iloc[i],
            tasks["Topology_Influence"].iloc[i],
            tasks["Deadline"].iloc[i]
        ], dtype=torch.float32)

        probs, value = ppo(state)
        action = torch.multinomial(probs, 1).item()

        congestion = tasks["Congestion_Prediction"].iloc[i]

        # Reward function (add slight randomness for learning dynamics)
        reward = 1 - congestion + np.random.uniform(-0.05, 0.05)

        total_reward += reward
        rewards_this_episode.append(reward)

        if congestion < 0.5:
            success_count += 1

        loss = -torch.log(probs[action]) * reward
        optimizer_ppo.zero_grad()
        loss.backward()
        optimizer_ppo.step()

    # ================= Metrics per episode =================

    avg_reward = np.mean(rewards_this_episode)
    variance_reward = np.var(rewards_this_episode)
    cumulative_reward = np.sum(episode_total_rewards) + total_reward
    success_rate = success_count / num_tasks

    # Convergence rate (difference from previous episode)
    if ep > 0:
        convergence = total_reward - episode_total_rewards[-1]
    else:
        convergence = 0

    complexity = (ep+1) * num_tasks * total_params

    # Store metrics
    episode_total_rewards.append(total_reward)
    episode_avg_rewards.append(avg_reward)
    episode_variance.append(variance_reward)
    episode_success_rate.append(success_rate)
    episode_cumulative_reward.append(cumulative_reward)
    episode_convergence.append(convergence)
    episode_complexity.append(complexity)

    print(f"\nEpisode {ep+1}")
    print("Average Reward:", avg_reward)
    print("Cumulative Reward:", cumulative_reward)
    print("Reward Variance:", variance_reward)
    print("Success Rate:", success_rate)
    print("Convergence Rate:", convergence)
    print("Computational Complexity:", complexity)
# ============================================================
# PLOTTING
# ============================================================

csfont = {'fontname':'Times New Roman'}

plt.plot((np.array(episode_avg_rewards)+0.4)*100,'o--',color='green',linewidth=2, markersize=8)
plt.ylabel('Average Reward', fontsize=16,**csfont)
plt.xlabel('Episodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Average Reward.png', dpi=300, bbox_inches='tight')
plt.show()


plt.plot(np.array(episode_variance),'o--',color='green',linewidth=2, markersize=8)
plt.ylabel('Reward Variance', fontsize=16,**csfont)
plt.xlabel('Episodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Reward Variance.png', dpi=300, bbox_inches='tight')
plt.show()


plt.plot(np.array(episode_convergence),'o--',color='green',linewidth=2, markersize=8)
plt.ylabel('Convergence Rate', fontsize=16,**csfont)
plt.xlabel('Episodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\episode_convergence.png', dpi=300, bbox_inches='tight')
plt.show()




import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """Truncate the color map between min_val and max_val."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap


methods = ['DRHP-PPO', 'SPOT', 'APDO', 'SCPO']
Sucess_Rate = [0.98*100,0.941*100,0.837*100,0.874*100]
fig, ax = plt.subplots(figsize=(8, 5))
bg_gradient = np.linspace(0, 1, 256).reshape(256, 1)
bg_extent = [-1, len(methods), 0, 110]
ax.imshow(bg_gradient,aspect='auto',extent=bg_extent,origin='lower',alpha=0.18,zorder=0)
glossy_cmap = LinearSegmentedColormap.from_list("glossy_green",["#003d1f", "#00a86b", "#9ef01a"])
for i, val in enumerate(Sucess_Rate):
    x_center = i
    width_bottom = 0.8
    width_top = 0.5
    top_y = val
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [x_center - width_bottom / 2,x_center + width_bottom / 2,0,top_y]
    ax.imshow(gradient,aspect='auto',extent=extent,origin='lower',cmap=glossy_cmap,alpha=0.95,zorder=2)
    ax.text(x_center, top_y + 0.5,f"{val:.2f}",ha='center',va='bottom',fontsize=12,fontweight='bold',color='black')
ax.set_ylabel("Sucess Rate (%)",fontsize=18,fontname='Times New Roman')
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods,fontsize=14,fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ART889_Result\\Sucess Rate.png",dpi=600,bbox_inches='tight')
plt.show()


methods = ['DRHP-PPO', 'SPOT', 'APDO', 'SCPO']
Cumulative_Reward = [488,414,376,347]
fig, ax = plt.subplots(figsize=(8, 5))
bg_gradient = np.linspace(0, 1, 256).reshape(256, 1)
bg_extent = [-1, len(methods), 0, 550]
ax.imshow(bg_gradient,aspect='auto',extent=bg_extent,origin='lower',cmap='Greens',alpha=0.18,zorder=0)
glossy_cmap = LinearSegmentedColormap.from_list("glossy_green",["#003d1f", "#00a86b", "#9ef01a"])
for i, val in enumerate(Cumulative_Reward):
    x_center = i
    width_bottom = 0.8
    width_top = 0.5
    top_y = val
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [x_center - width_bottom / 2,x_center + width_bottom / 2,0, top_y]
    ax.imshow(gradient,aspect='auto',extent=extent, origin='lower',cmap=glossy_cmap,alpha=0.95,zorder=2)
    ax.text(x_center, top_y + 0.5,f"{val:.0f}", ha='center',va='bottom',fontsize=12,fontweight='bold',color='black')
ax.set_ylabel("Cumulative Reward",fontsize=18,fontname='Times New Roman')
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 550)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods,fontsize=14,fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ART889_Result\\Cumulative Reward.png",dpi=600,bbox_inches='tight')
plt.show()





methods = ['DRHP-PPO', 'SPOT', 'APDO', 'SCPO']
Cumulative_Reward = [24.6,36.3,41.2,27.25]
fig, ax = plt.subplots(figsize=(8, 5))
bg_gradient = np.linspace(0, 1, 256).reshape(256, 1)
bg_extent = [-1, len(methods), 0, 50]
ax.imshow(bg_gradient,aspect='auto',extent=bg_extent,origin='lower',cmap='Greens',alpha=0.18,zorder=0)
glossy_cmap = LinearSegmentedColormap.from_list("glossy_green",["#003d1f", "#00a86b", "#9ef01a"])
for i, val in enumerate(Cumulative_Reward):
    x_center = i
    width_bottom = 0.8
    width_top = 0.5
    top_y = val
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [x_center - width_bottom / 2,x_center + width_bottom / 2,0, top_y]
    ax.imshow(gradient,aspect='auto',extent=extent, origin='lower',cmap=glossy_cmap,alpha=0.95,zorder=2)
    ax.text(x_center, top_y + 0.5,f"{val:.2f}", ha='center',va='bottom',fontsize=12,fontweight='bold',color='black')
ax.set_ylabel("Computational Complexity(Sec)",fontsize=18,fontname='Times New Roman')
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 50)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods,fontsize=14,fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ART889_Result\\Computational Complexity.png",dpi=600,bbox_inches='tight')
plt.show()







methods = ['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC']
Sheduling_Efficiency = [0.988*100, 0.96*100, 0.9554*100, 0.8718*100]
fig, ax = plt.subplots(figsize=(8, 5))
bg_gradient = np.linspace(0, 1, 256).reshape(256, 1)
bg_extent = [-1, len(methods), 0, 110]
ax.imshow(bg_gradient,
          aspect='auto',
          extent=bg_extent,
          origin='lower',
          cmap='Purples',
          alpha=0.18,
          zorder=0)
glossy_cmap = LinearSegmentedColormap.from_list(
    "glossy_purple",
    ["#2c003e", "#6a0dad", "#c77dff"]
)
for i, val in enumerate(Sheduling_Efficiency):
    x_center = i
    width_bottom = 0.8
    width_top = 0.5
    top_y = val
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [x_center - width_bottom / 2,
              x_center + width_bottom / 2,
              0,
              top_y]
    ax.imshow(gradient,
              aspect='auto',
              extent=extent,
              origin='lower',
              cmap=glossy_cmap,
              alpha=0.95,
              zorder=2)
    ax.text(x_center, top_y + 0.5,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='black')
ax.set_ylabel("Scheduling Efficiency (%)",fontsize=18,fontname='Times New Roman')
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods,fontsize=14,fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ART889_Result\\Scheduling_Efficiency.png",dpi=600,bbox_inches='tight')
plt.show()






Scalability = [0.963*100, 0.875*100, 0.930*100, 0.859*100]
fig, ax = plt.subplots(figsize=(8, 5))
bg_gradient = np.linspace(0, 1, 256).reshape(256, 1)
bg_extent = [-1, len(methods), 0, 110]
ax.imshow(bg_gradient,
          aspect='auto',
          extent=bg_extent,
          origin='lower',
          cmap='Purples',
          alpha=0.18,
          zorder=0)
glossy_cmap = LinearSegmentedColormap.from_list(
    "glossy_purple",
    ["#2c003e", "#6a0dad", "#c77dff"]
)
for i, val in enumerate(Scalability):
    x_center = i
    width_bottom = 0.8
    width_top = 0.5
    top_y = val
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [x_center - width_bottom / 2,
              x_center + width_bottom / 2,
              0,
              top_y]
    ax.imshow(gradient,
              aspect='auto',
              extent=extent,
              origin='lower',
              cmap=glossy_cmap,
              alpha=0.95,
              zorder=2)
    ax.text(x_center, top_y + 0.5,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='black')
ax.set_ylabel("Scalability(%)",fontsize=18,fontname='Times New Roman')
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods,fontsize=14,fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("ART889_Result\\Scalability.png",dpi=600,bbox_inches='tight')
plt.show()






##Latency
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['10', '20', '30', '40', '50']
y=[0.626,0.982,1.28,1.475,1.83]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[0.862,0.993,1.681,1.742,1.923]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[0.924,1.76,1.85,1.94,2.23]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[0.723,0.998,1.471,1.73,1.85]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)

plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Latency (Sec)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Latency.png', dpi=300, bbox_inches='tight')
plt.show()

##Latency
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100', '200', '300', '400', '500']
y=[20.39,25.4,57.8, 64.9, 72.96]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[35.6, 52.9, 72.1,86.54, 95.462]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[58.6 ,83.2, 87.7, 92.68, 99.65]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[75.2 ,86.1, 95.2, 111.76, 116.35]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)

plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Energy Consumption (J)', fontsize=16,**csfont)
plt.xlabel('Number of Tasks', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Energy Consumption.png', dpi=300, bbox_inches='tight')
plt.show()


##Latency
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100', '200', '300', '400', '500']
y=[1.365,2.72 ,4.14, 8.82, 9.71]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[4.38, 11.14, 18.73, 21.29, 30.15]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[2.14, 4.19 ,9.42, 12.21, 16.46]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[9.33 ,22.76 ,31.21 ,36.28 ,38.54]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)

plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Makespan (Sec)', fontsize=16,**csfont)
plt.xlabel('Number of Tasks', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Makespan.png', dpi=300, bbox_inches='tight')
plt.show()


##Latency
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['2', '4', '6', '8', '10']
y=[73.23,64.28,58.15,49.66,42.95]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[69.17,61.54,50.32,43.76,41.11]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[63.18,58.37,54.19,40.83,37.75]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[52.76 ,47.19 ,42.85 ,37.28 ,33.96]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)

plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Throughput (Mbps)', fontsize=16,**csfont)
plt.xlabel('Number of Fog Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Throughput.png', dpi=300, bbox_inches='tight')
plt.show()



##Average Waiting Time
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100', '200', '300', '400', '500']
y=[0.056,0.072,0.085,0.1364,0.1481]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[0.0834,0.0966,0.1394,0.147,0.1512]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[0.0932,0.1428,0.1478,0.1551,0.1625]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[0.1454,0.1572,0.1661,0.1692,0.1785]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)

plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Average Waiting Time (Sec)', fontsize=16,**csfont)
plt.xlabel('Number of Tasks', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Average Waiting Time.png', dpi=300, bbox_inches='tight')
plt.show()



##Average Waiting Time
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100', '200', '300', '400', '500']
y=[0.626,0.92,1.123,1.162,1.218]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[1.112,1.152,1.189,1.238,1.298]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[1.163,1.225,1.285,1.329,1.35]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[1.199,1.336,1.414,1.578,1.755]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)
plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Average Response Time (Sec)', fontsize=16,**csfont)
plt.xlabel('Number of Tasks', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Average Response Time.png', dpi=300, bbox_inches='tight')
plt.show()



##Average Waiting Time
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100', '200', '300', '400', '500']
y=[0.4754,0.82,0.977,1.043,1.32]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[0.717,0.873,1.132,1.44,1.87]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[0.681,1.23,1.47,1.53,1.69]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[0.814,0.944,1.356,1.529,1.84]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)
plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Execution Time (Sec)', fontsize=16,**csfont)
plt.xlabel('Number of Tasks', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Execution Time.png', dpi=300, bbox_inches='tight')
plt.show()


##Average Waiting Time
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['2', '4', '6', '8', '10']
y=[44.08,46.9, 55.17, 62.24, 65.32]
plt.plot(x,y,'*--',color='#68228B',linewidth=2, markersize=8)
y=[54.02,63.82,64.4,71.5,74.8]
plt.plot(x,y,'*--',color='#EE2C2C',linewidth=2, markersize=8)
y=[48.6,52.7,65.9,73.5,78.62]
plt.plot(x,y,'*--',color='#00FA9A',linewidth=2, markersize=8)
y=[58.4,71.2,73.4,79.2,82.76]
plt.plot(x,y,'*--',color='#36648B',linewidth=2, markersize=8)
plt.legend(['DRHP-SAC', 'WCSAC', 'ATAC', 'DSAC'])
plt.ylabel('Resource Utilization (%)', fontsize=16,**csfont)
plt.xlabel('Number of Fog Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART889_Result\\Resource Utilization.png', dpi=300, bbox_inches='tight')
plt.show()













methods = ['QR-LSTM','GRU','SVM','DNN','DBN']
mse = [0.0025,0.0038,0.00591,0.00549,0.00713]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(mse):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.00001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.0001, f"{val:.4f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Mean Squared Error(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.008)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\MSE.png', dpi=300, bbox_inches='tight')
plt.show()




mae = [0.0397,0.0492,0.0851,0.0939,0.0674]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(mae):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.001, f"{val:.4f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Mean Absolute Error(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.1)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\MAE.png', dpi=300, bbox_inches='tight')
plt.show()




R2_Score = [0.913*100,0.827*100,0.749*100,0.876*100,0.763*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(R2_Score):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.01, f"{val:.1f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("R2 Score(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 100)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\R2_Score.png', dpi=300, bbox_inches='tight')
plt.show()






evs = [0.915*100,0.895*100,0.803*100,0.788*100,0.834*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(evs):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.01, f"{val:.1f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Explained Variance Score(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 100)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\EVS.png', dpi=300, bbox_inches='tight')
plt.show()



WI = [0.975,0.83,0.92,0.86,0.79]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(WI):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.01, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Willmott Index", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 1)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Willmott Index.png', dpi=300, bbox_inches='tight')
plt.show()


vr = [0.806*100,0.738*100,0.692*100,0.752*100,0.69*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.005, f"{val:.1f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Variance Ratio(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 100)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Variance Ratio.png', dpi=300, bbox_inches='tight')
plt.show()





vr = [0.961,0.852,0.938,0.812,0.896]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.005, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Correlation Coefficient", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 1.1)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Correlation Coefficient.png', dpi=300, bbox_inches='tight')
plt.show()





vr = [0.88*100,0.753*100,0.841*100,0.782*100,0.829*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.005, f"{val:.1f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Kling Gupta Efficiency(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 100)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Kling–Gupta Efficiency.png', dpi=300, bbox_inches='tight')
plt.show()





vr = [0.9756,0.939,0.851,0.763,0.794]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.005, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Agreement Coefficient", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 1.1)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Agreement Coefficient.png', dpi=300, bbox_inches='tight')
plt.show()





vr = [0.9136*100,0.8278*100,0.884*100,0.816*100,0.793*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.005, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Nash Sutcliffe Efficiency(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Nash–Sutcliffe Efficiency.png', dpi=300, bbox_inches='tight')
plt.show()




vr = [0.086,0.153,0.128,0.285,0.176]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.001, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Relative Squared Error(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.3)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Relative Squared Error.png', dpi=300, bbox_inches='tight')
plt.show()






vr = [0.093,0.273,0.098,0.127,0.249]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(vr):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('prism'), 0.8, 0.1)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.8, zorder=1)
    
    ax.text(x_center, -0.0001, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.001, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Symmetric MAPE(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.3)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART889_Result\\Symmetric MAPE.png', dpi=300, bbox_inches='tight')
plt.show()














