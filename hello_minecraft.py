# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------
# Install the Malmo environment.
# Set the environment such as:
#   MALMO=~/Malmo-0.18.0-Linux-Ubuntu-14.04-64bit
#   export MALMO_XSD_PATH=$MALMO/Schemas
#   export PYTHONPATH=$MALMO/Python_Examples:$PYTHONPATH

import MalmoPython
import os
import sys
import time
import numpy as np

agent_host = MalmoPython.AgentHost()
my_mission = MalmoPython.MissionSpec()
my_mission_record = MalmoPython.MissionRecordSpec()

my_mission.timeLimitInSeconds(1000)
my_mission.createDefaultTerrain()
my_mission.forceWorldReset()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print "Error starting mission:",e
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print "Waiting for the mission to start ",
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print "Error:",error.text

print
print "Mission running ",

actions = ["move", "strafe", "pitch", "turn", "jump", "crouch", "attack"]
degrees = [-1, 0, 1]

while world_state.is_mission_running:
	time.sleep(1)
	world_state = agent_host.getWorldState()

	action = np.random.choice(a=actions, size=1)[0]
	degree = np.random.choice(a=degrees, size=1)[0]
	cmd = '{} {}'.format(action, degree)
	print cmd
	agent_host.sendCommand(cmd)

print 'Your world has ended'