import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
import re

# Page Config (Wide mode for better grid viewing)
st.set_page_config(layout="wide", page_title="MPPI Debugger")

# --- 1. Data Loading & Scanning ---
@st.cache_data(show_spinner=True)
def load_log_data(path, step, ep):
    full_path = os.path.join(path, f'plans_step_{step}_ep_{ep}.pt')
    data = torch.load(full_path, map_location='cpu')
    return data['frames'], data['traj_plans']

@st.cache_data
def find_available_logs(root_dir):
    # Find all 'eval_trajectories' folders recursively
    if not os.path.exists(root_dir): return []
    return sorted(glob.glob(os.path.join(root_dir, '**', 'eval_trajectories'), recursive=True))

@st.cache_data
def get_steps_and_episodes(log_path):
    pattern = re.compile(r'plans_step_(\d+)_ep_(\d+)\.pt')
    files = glob.glob(os.path.join(log_path, 'plans_step_*_ep_*.pt'))
    step_ep_map = {}
    
    for f in files:
        match = pattern.search(f)
        if match:
            step = int(match.group(1))
            ep = int(match.group(2))
            if step not in step_ep_map: step_ep_map[step] = []
            step_ep_map[step].append(ep)
            
    steps = sorted(step_ep_map.keys())
    for s in steps: step_ep_map[s] = sorted(step_ep_map[s])
    return steps, step_ep_map

# --- 2. Sidebar Controls ---
st.sidebar.title("MPPI Controls")
search_root = st.sidebar.text_input("Search Root", value='logs')

available_logs = find_available_logs(search_root)
if available_logs:
    log_path = st.sidebar.selectbox("Log Path", available_logs)
else:
    st.sidebar.warning(f"No 'eval_trajectories' found in {search_root}")
    log_path = None

step_num = None
ep_num = None

if log_path:
    avail_steps, step_ep_map = get_steps_and_episodes(log_path)
    if avail_steps:
        step_num = st.sidebar.selectbox("Step Number", avail_steps, index=len(avail_steps)-1)
        avail_eps = step_ep_map[step_num]
        ep_num = st.sidebar.selectbox("Episode Number", avail_eps)
    else:
        st.sidebar.warning("No plan files found in selected log path.")

if st.sidebar.button("Load Data"):
    if log_path and step_num is not None and ep_num is not None:
        try:
            frames, plans = load_log_data(log_path, step_num, ep_num)
            st.session_state['frames'] = frames
            st.session_state['plans'] = plans
            
            # Generate video volume for animation (H, W, C)
            video_frames = []
            for img in frames:
                if isinstance(img, torch.Tensor):
                    if img.shape[0] == 3: img = img.permute(1, 2, 0)
                    img = img.cpu().numpy()
                else: img = np.array(img)
                # Ensure 0-255 uint8 for bandwidth efficiency
                if img.max() <= 1.0: img = img * 255.0
                video_frames.append(img.astype(np.uint8))
            st.session_state['video_vol'] = np.stack(video_frames)
            
            st.session_state['data_loaded'] = True
            st.sidebar.success(f"Loaded Step {step_num} Ep {ep_num} ({len(frames)} frames)!")
        except Exception as e:
            st.sidebar.error(f"Error loading: {e}")
    else:
        st.sidebar.error("Please select a valid log, step and episode.")

# --- 3. Main Interface ---
if st.session_state.get('data_loaded'):
    frames = st.session_state['frames']
    plans = st.session_state['plans']
    
    # Layout: Control/Obs (Left) | Analysis (Right)
    col_ctrl, col_viz = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Control")
        
        # 1. Animation Player (Fast, client-side scrubbing)
        if 'video_vol' in st.session_state:
            st.caption("Episode Playback (Smooth)")
            # Animation frame=0 allows plotly to handle the slider client-side
            fig_anim = px.imshow(st.session_state['video_vol'], animation_frame=0, binary_string=True, height=300)
            fig_anim.update_xaxes(showticklabels=False, title=None)
            fig_anim.update_yaxes(showticklabels=False, title=None)
            fig_anim.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            # Sane default speed
            fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
            st.plotly_chart(fig_anim, use_container_width=True)

        st.divider()

        # 2. Analysis Slider (Server-side, triggers heavy plots)
        st.subheader("Analysis Selection")
        frame_idx = st.slider("Select Step to Analyze", 0, len(frames)-1, 0)
        
        # Display Current Frame (Static check)
        current_plan = plans[frame_idx]['plan']

        # Iteration Stats
        st.markdown("#### Iteration Values")
        # Helper to get scalar safely
        def get_scalar(val):
            if torch.is_tensor(val): return val.item()
            return val
            
        val_data = {f"Iter {i}": [f"{get_scalar(d['elite_values'].mean()):.2f}"] for i, d in enumerate(current_plan)}
        st.dataframe(val_data, hide_index=True)

    with col_viz:
        # --- Reward Analysis ---
        if plans and 'reward' in plans[0]:
            # Extract and ensure values are floats (handle tensors if present)
            r_true = [p['reward'].item() if torch.is_tensor(p['reward']) else p['reward'] for p in plans]
            r_pred = [p.get('pred_reward', 0).item() if torch.is_tensor(p.get('pred_reward', 0)) else p.get('pred_reward', 0) for p in plans]
            
            fig_rew = go.Figure()
            fig_rew.add_trace(go.Scatter(y=r_true, mode='lines', name='True Reward'))
            fig_rew.add_trace(go.Scatter(y=r_pred, mode='lines', name='Pred Reward'))
            
            # Add indicator for current timestep
            fig_rew.add_vline(x=frame_idx, line_width=2, line_dash="dash", line_color="green", annotation_text="Current")
            
            fig_rew.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rew, use_container_width=True)

        # --- B. Display Plotly Grid ---
        st.subheader("MPPI Optimization Landscape")
        
        num_iters = len(current_plan)
        horizon = current_plan[0]['mean'].shape[0]
        
        # Create Subplots
        fig = make_subplots(
            rows=horizon, cols=num_iters,
            shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.01, vertical_spacing=0.01,
            column_titles=[f"Iter {i}" for i in range(num_iters)],
            row_titles=[f"t={t}" for t in range(horizon)]
        )

        for col_idx, data in enumerate(current_plan):
            # Extract
            elite_actions = data['elite_actions'].detach().cpu().numpy()
            scores = data['score'].detach().cpu().numpy().flatten()
            mean = data['mean'].detach().cpu().numpy()
            std = data['std'].detach().cpu().numpy()
            
            # Rewards
            e_idxs = data['elite_idxs'].long().cpu()
            all_rews = data['rewards'].detach().cpu()
            elite_rews = all_rews[:, e_idxs, 0].numpy()
            
            # Q-values & Values
            all_qs = data['q_values'].detach().cpu()
            elite_qs = all_qs[e_idxs, 0].numpy()

            all_vals = data['values'].detach().cpu()
            elite_vals = all_vals[e_idxs, 0].numpy()

            for t in range(horizon):
                # 1. Elites
                hover_text = [
                    f"<b>Idx:</b> {i}<br><b>Reward:</b> {elite_rews[t, i]:.4f}<br><b>Q:</b> {elite_qs[i]:.4f}<br><b>Value:</b> {elite_vals[i]:.4f}<br><b>Weight:</b> {scores[i]:.4f}" 
                    for i in range(len(scores))
                ]
                
                fig.add_trace(
                    go.Scattergl( # Use WebGL for speed
                        x=elite_actions[t, :, 0],
                        y=elite_actions[t, :, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=scores,
                            colorscale='Plasma',
                            showscale=(t==0 and col_idx==num_iters-1), # Only show legend once
                            line=dict(width=1, color='Black')
                        ),
                        text=hover_text,
                        hoverinfo='text',
                        name='Elites'
                    ),
                    row=t+1, col=col_idx+1
                )
                
                # 2. Mean
                fig.add_trace(
                    go.Scattergl(
                        x=[mean[t, 0]], y=[mean[t, 1]],
                        mode='markers',
                        marker=dict(symbol='x', size=10, color='red'),
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=t+1, col=col_idx+1
                )
                
                # 3. Std (Ellipse)
                fig.add_shape(
                    type="circle",
                    xref=f"x{col_idx+1 + (t*num_iters) if t>0 else col_idx+1}", # Tricky plotly referencing
                    yref=f"y{col_idx+1 + (t*num_iters) if t>0 else col_idx+1}",
                    x0=mean[t, 0] - std[t, 0]*2,
                    y0=mean[t, 1] - std[t, 1]*2,
                    x1=mean[t, 0] + std[t, 0]*2,
                    y1=mean[t, 1] + std[t, 1]*2,
                    line_color="red", line_dash="dash",
                    row=t+1, col=col_idx+1
                )

        fig.update_layout(
            height=horizon * 250, # Dynamic height
            margin=dict(l=40, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        fig.update_xaxes(range=[-1.2, 1.2], showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(range=[-1.2, 1.2], showgrid=True, gridcolor='lightgray')

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter path and click 'Load Data' in the sidebar.")