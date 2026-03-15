#!/bin/bash
SESSION="o_session" # lol obsession with o's

# check if gflow is running and if not start it
output="$(gqueue 2>&1 >/dev/null)"
if echo "$output" | grep -q "Could not connect to gflowd server"; then
    echo "Starting gflow..."
    gflowd up
else
    echo "gflow is already running."
fi

# needs to be fixed!

# # Check if the session already exists
# tmux has-session -t $SESSION 2>/dev/null

# if [ $? != 0 ]; then
#   # 1. Create session (Pane 0 created automatically)
#   tmux new-session -d -s $SESSION -n "Main"
  
#   # 2. Split Pane 0 horizontally (Creates Pane 1 on the right)
#   tmux split-window -h -t $SESSION:0.0
  
#   # 3. Split the Left Pane (0) vertically (Creates Pane 2)
#   tmux split-window -v -t $SESSION:0.0
  
#   # 4. Split the Right Pane (1) vertically (Creates Pane 3)
#   # Note: After the previous split, the right pane is now index 2
#   tmux split-window -v -t $SESSION:0.2

#   # --- Send Commands ---
  
#   # Top Left (Pane 0)
#   tmux send-keys -t $SESSION:0.0 "../../.venv/bin/python full_client.py" C-m

#   # Bottom Left (Pane 1)
#   tmux send-keys -t $SESSION:0.2 "CUDA_VISIBLE_DEVICES=0 /home/cavadalab/Documents/scsv/the_thing/backend/models/segmentation/sam3/.venv/bin/python worker_sam.py" C-m

#   # Top Right (Pane 2)
#   tmux send-keys -t $SESSION:0.1 "/home/cavadalab/Documents/scsv/covision/full_pipeline/testing_zeromq/.venv/bin/python broker.py" C-m

#   # Bottom Right (Pane 3)
#   tmux send-keys -t $SESSION:0.3 "CUDA_VISIBLE_DEVICES=0,1 /home/cavadalab/Documents/scsv/the_thing/backend/models/vlm/internVL/.venv/bin/python worker_vlm.py" C-m

#   tmux select-pane -t $SESSION:0.0
# fi

# # Attach to the session
# tmux attach-session -t $SESSION

# gflowd down